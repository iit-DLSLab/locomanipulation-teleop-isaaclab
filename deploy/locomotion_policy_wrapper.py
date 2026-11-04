# Description: Wrapper of the locomotion policy

# Authors:
# Giulio Turrisi

import time
import copy
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from tqdm import tqdm
import mujoco
import onnxruntime as ort
import torch

import config

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../source/locomanipulation_teleop_isaaclab/locomanipulation_teleop_isaaclab/tasks/")
from supervised_learning_networks import load_network

class LocomotionPolicyWrapper:
    def __init__(self, mjModel):

        self.policy = ort.InferenceSession(config.locomotion_policy_folder_path + "/exported/policy.onnx")
        self.Kp_walking = config.Kp_walking
        self.Kd_walking = config.Kd_walking
        self.Kp_stand_up_and_down = config.Kp_stand_up_and_down
        self.Kd_stand_up_and_down = config.Kd_stand_up_and_down

        self.RL_FREQ = 1./(config.training_env["sim"]["dt"]*config.training_env["decimation"])  # Hz, frequency of the RL controller


        # Max change of the velocity command per second (m/s^2 and rad/s^2), so a step from keyboard or joystick becomes a ramp
        self.max_cmd_acc = 1.0
        self.filtered_cmd = np.zeros(3)  # vx, vy in the heading frame and yaw rate

        # RL controller initialization -------------------------------------------------------------
        self.action_scale = config.training_env["action_scale"]
        self.past_rl_actions = np.zeros(12) # only legs
        

        keyframe_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        standUp_qpos = mjModel.key_qpos[keyframe_id]
        self.default_joint_pos_leg = standUp_qpos[7:19]

        self.default_joint_pos_arm = standUp_qpos[19:25]


        # Observation space initialization -------------------------------------------------------
        self.use_clock_signal = config.training_env["use_clock_signal"]


        # Step frequency ramps linearly with the commanded xy linear velocity norm, from desired_step_freq
        # (at/below step_freq_vel_norm_low) up to desired_step_freq_max (at/above step_freq_vel_norm_high)
        self.desired_step_freq = config.training_env["desired_step_freq"]
        self.desired_step_freq_max = config.training_env["desired_step_freq_max"]
        self.step_freq_vel_norm_low = config.training_env["step_freq_vel_norm_low"]
        self.step_freq_vel_norm_high = config.training_env["step_freq_vel_norm_high"]
        self.step_freq = self.desired_step_freq
        self.duty_factor = config.training_env["desired_duty_factor"]
        self.phase_offset = np.array(config.training_env["desired_phase_offset"])
        self.phase_signal = self.phase_offset.copy()


        self.desired_clip_actions = config.training_env["desired_clip_actions"]


        self.use_filter_actions = config.training_env["use_filter_actions"]
        self.action_filter_alpha = config.training_env["action_filter_alpha"]
        self.filtered_rl_actions = np.zeros(12) # only legs


        self.use_observation_history = config.training_env["use_observation_history"]
        if(self.use_observation_history):
            single_observation_space = config.training_env["single_observation_space"]
            self.history_length = config.training_env["history_length"]
            self._observation_history = np.zeros((self.history_length, single_observation_space), dtype=np.float32)


        # vision policies are the ones trained with the perceptive height scanner
        self.use_vision = config.training_env.get("perceptive_height_scanner") is not None

        # Learned State Estimator
        self.base_state_source = config.training_env["base_state_source"]
        if(self.base_state_source == "concurrent_se"):
            self._concurrent_state_est_network = load_network(config.concurrent_state_est_network, device='cpu')
            single_concurrent_state_est_observation_space = int(config.training_env["single_concurrent_state_est_observation_space"])
            concurrent_state_est_history_length = int(config.training_env["concurrent_state_est_history_length"])
            self._observation_history_concurrent_state_est = np.zeros((concurrent_state_est_history_length, single_concurrent_state_est_observation_space), dtype=np.float32)


        # Desired joint vector
        self.desired_joint_pos = np.zeros(12) # only legs


    def _get_projected_gravity(self, quat_wxyz):        
        # Get the projected gravity in the base frame
        GRAVITY_VEC_W = torch.tensor((0, 0, -9.81), dtype=torch.double)
        GRAVITY_VEC_W = GRAVITY_VEC_W / GRAVITY_VEC_W.norm(p=2, dim=-1).clamp(min=1e-9, max=None).unsqueeze(-1)
        q = torch.tensor(quat_wxyz).view(1, 4)
        v = GRAVITY_VEC_W.clone().detach().view(1, 3)
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        # for two-dimensional tensors, bmm is faster than einsum
        if q_vec.dim() == 2:
            c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
        else:
            c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
        projected_gravity =  a - b + c
        return projected_gravity.numpy().flatten()


    def compute_control(self, 
            base_pos, 
            base_ori_euler_xyz, 
            base_quat_wxyz,
            base_lin_vel, 
            base_ang_vel, 
            heading_orientation_SO3,
            joints_pos_leg, 
            joints_vel_leg,
            ref_base_lin_vel, 
            ref_base_ang_vel,
            ref_pose_command,
            joints_pos_arm,
            imu_linear_acceleration=None,
            imu_angular_velocity=None,
            imu_orientation=None,
            heightmap_data=None):


        # Update Observation ----------------------        
        if(self.base_state_source != "ground_truth"):
            base_projected_gravity = self._get_projected_gravity(imu_orientation)
            # The IMU linear acceleration is not used: zeroed with "imu" as in training, replaced by the concurrent SE
            # prediction with "concurrent_se"
            base_linear = np.zeros(3)
            base_ang_vel = imu_angular_velocity
        else:
            base_projected_gravity = self._get_projected_gravity(base_quat_wxyz)
            base_linear = base_lin_vel
            base_ang_vel = base_ang_vel

        
        # Get the reference base velocity in the world frame
        ref_base_lin_vel_h = heading_orientation_SO3.T@ref_base_lin_vel

        # Rate limit the command
        target_cmd = np.array([ref_base_lin_vel_h[0], ref_base_lin_vel_h[1], ref_base_ang_vel[2]])
        max_delta = self.max_cmd_acc / self.RL_FREQ
        self.filtered_cmd += np.clip(target_cmd - self.filtered_cmd, -max_delta, max_delta)
        ref_base_lin_vel_h[0:2] = self.filtered_cmd[0:2]
        ref_base_ang_vel = np.array([ref_base_ang_vel[0], ref_base_ang_vel[1], self.filtered_cmd[2]])
        
            
        # Fill the observation vector
        joints_pos_delta = joints_pos_leg - self.default_joint_pos_leg
        joints_pos_delta_FL = joints_pos_delta[0:3]
        joints_pos_delta_FR = joints_pos_delta[3:6]
        joints_pos_delta_RL = joints_pos_delta[6:9]
        joints_pos_delta_RR = joints_pos_delta[9:12]

        joints_vel_FL = joints_vel_leg[0:3]
        joints_vel_FR = joints_vel_leg[3:6]
        joints_vel_RL = joints_vel_leg[6:9]
        joints_vel_RR = joints_vel_leg[9:12]

        joints_pos_arm_delta = joints_pos_arm - self.default_joint_pos_arm

        # Phase Signal
        if(self.use_clock_signal):
            # Ramp the step frequency linearly with the commanded xy linear velocity norm, as in training
            ramp = (np.linalg.norm(ref_base_lin_vel_h[0:2]) - self.step_freq_vel_norm_low) / (
                self.step_freq_vel_norm_high - self.step_freq_vel_norm_low
            )
            ramp = np.clip(ramp, 0.0, 1.0)
            self.step_freq = self.desired_step_freq + ramp * (self.desired_step_freq_max - self.desired_step_freq)

            self.phase_signal += self.step_freq * (1 / (self.RL_FREQ))
            self.phase_signal = self.phase_signal % 1.0
            # While standing the phase is held at the phase offset, as in training
            commands = np.array([ref_base_lin_vel_h[0], ref_base_lin_vel_h[1], ref_base_ang_vel[2]], dtype=np.float32)
            if(np.linalg.norm(commands) < 0.01):
                self.phase_signal = self.phase_offset.copy()
            # sin and cos of the phase of each leg, as in training: [sin FL..RR, cos FL..RR]
            clock_data = np.concatenate((np.sin(2.0 * np.pi * self.phase_signal), np.cos(2.0 * np.pi * self.phase_signal))).astype(np.float32)


        if(self.base_state_source == "concurrent_se"):
            # as in training, the concurrent SE does not use the IMU linear acceleration
            obs_concurrent_state_est = np.concatenate([
                base_ang_vel, 
                base_projected_gravity,
                ref_base_lin_vel_h[0:2],
                [ref_base_ang_vel[2]],
                ref_pose_command,
                [joints_pos_delta_FL[0]], [joints_pos_delta_FR[0]], [joints_pos_delta_RL[0]], [joints_pos_delta_RR[0]],
                [joints_pos_delta_FL[1]], [joints_pos_delta_FR[1]], [joints_pos_delta_RL[1]], [joints_pos_delta_RR[1]],
                [joints_pos_delta_FL[2]], [joints_pos_delta_FR[2]], [joints_pos_delta_RL[2]], [joints_pos_delta_RR[2]],
                [joints_vel_FL[0]], 
                [joints_vel_FR[0]], 
                [joints_vel_RL[0]], 
                [joints_vel_RR[0]],

                [joints_vel_FL[1]],
                [joints_vel_FR[1]], 
                [joints_vel_RL[1]], 
                [joints_vel_RR[1]],
                
                [joints_vel_FL[2]], 
                [joints_vel_FR[2]], 
                [joints_vel_RL[2]], 
                [joints_vel_RR[2]],
                self.past_rl_actions.copy(),
                joints_pos_arm_delta
            ])
            #the bottom element is the newest observation!!
            past_concurrent_state_est = self._observation_history_concurrent_state_est[1:,:]
            self._observation_history_concurrent_state_est = np.vstack((past_concurrent_state_est, copy.deepcopy(obs_concurrent_state_est)))
            obs_concurrent_state_est = self._observation_history_concurrent_state_est.flatten()
            
            # QUERY THE NETOWRK
            base_linear = self._concurrent_state_est_network(torch.tensor(obs_concurrent_state_est, dtype=torch.float32).unsqueeze(0)).detach().numpy().squeeze()
            #base_linear = base_lin_vel
        
        
        obs = np.concatenate([
            base_linear, # zeros with base_state_source "imu", linear vel from the concurrent SE or the ground truth otherwise
            base_ang_vel, # imu angular vel if base_state_source is not "ground_truth"
            base_projected_gravity,
            ref_base_lin_vel_h[0:2],
            [ref_base_ang_vel[2]],
            ref_pose_command,
            [joints_pos_delta_FL[0]], [joints_pos_delta_FR[0]], [joints_pos_delta_RL[0]], [joints_pos_delta_RR[0]],
            [joints_pos_delta_FL[1]], [joints_pos_delta_FR[1]], [joints_pos_delta_RL[1]], [joints_pos_delta_RR[1]],
            [joints_pos_delta_FL[2]], [joints_pos_delta_FR[2]], [joints_pos_delta_RL[2]], [joints_pos_delta_RR[2]],
            [joints_vel_FL[0]], 
            [joints_vel_FR[0]], 
            [joints_vel_RL[0]], 
            [joints_vel_RR[0]],

            [joints_vel_FL[1]], 
            [joints_vel_FR[1]], 
            [joints_vel_RL[1]], 
            [joints_vel_RR[1]],
            
            [joints_vel_FL[2]], 
            [joints_vel_FR[2]], 
            [joints_vel_RL[2]], 
            [joints_vel_RR[2]],
            self.past_rl_actions.copy(),
            clock_data if self.use_clock_signal else np.array([]),
            joints_pos_arm_delta,
        ])


        if(self.use_observation_history):
            #the bottom element is the newest observation!!
            past = self._observation_history[1:,:]
            self._observation_history = np.vstack((past, copy.deepcopy(obs)))
            obs = self._observation_history.flatten()

        
        if(self.use_vision):
            # Flatten heightmap with bottom-right at [0], then points going upward
            heightmap_2d = heightmap_data[..., 2][:, :, 0]  # Remove the last dimension
            
            # Flip vertically (so bottom row becomes first) and horizontally (so rightmost becomes first)
            heightmap_flipped = np.flip(heightmap_2d, axis=(0, 1))
            
            # Flatten column-wise so bottom-right is [0], then element above it is [1], etc.
            heightmap_data_isaac_convention = heightmap_flipped.flatten(order='F')

            height_data = (base_pos[2] - heightmap_data_isaac_convention - 0.5)
            height_data = height_data.clip(-1.0, 1.0)
            obs = np.concatenate((obs, height_data), axis=0)
            
        
        # RL Prediction
        obs = obs.reshape(1, -1)
        obs = obs.astype(np.float32)
        rl_action_temp = self.policy.run(None, {'obs': obs})[0][0]
        rl_action_temp = np.clip(rl_action_temp, -self.desired_clip_actions, self.desired_clip_actions)
        

        # Action Filtering
        # The raw action is the last action in the observation, as in training
        self.past_rl_actions = rl_action_temp.copy()
        if(self.use_filter_actions):
            # Exponential moving average (low-pass on the filter output), as in training
            alpha = self.action_filter_alpha
            self.filtered_rl_actions = alpha * rl_action_temp + (1-alpha) * self.filtered_rl_actions
            rl_action_temp = self.filtered_rl_actions.copy()

        # Use this print to understand the order of joints in the locomotion policy
        # first HIP, then THIGH, then CALF for each leg
        # but mujoco expect per leg order FL, FR, RL, RR for each joint
        #print("order of policy joint:", config.training_env["desired_joints_order"])
        rl_action = np.array([
            rl_action_temp[0], rl_action_temp[4], rl_action_temp[8],
            rl_action_temp[1], rl_action_temp[5], rl_action_temp[9],
            rl_action_temp[2], rl_action_temp[6], rl_action_temp[10],
            rl_action_temp[3], rl_action_temp[7], rl_action_temp[11],
        ])

        # Impedence Loop
        #TODO fix order of joints in locomotion env
        self.desired_joint_pos = self.default_joint_pos_leg + rl_action*self.action_scale

        
        return self.desired_joint_pos

