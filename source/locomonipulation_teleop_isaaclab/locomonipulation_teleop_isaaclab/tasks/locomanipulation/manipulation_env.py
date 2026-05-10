# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import onnxruntime as ort

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns, Imu
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from isaaclab.markers import VisualizationMarkers

from .manipulation_env_cfg import ManipulationFlatEnvCfg, ManipulationRoughBlindEnvCfg, ManipulationRoughVisionEnvCfg

class ManipulationEnv(DirectRLEnv):
    cfg: ManipulationFlatEnvCfg | ManipulationRoughBlindEnvCfg | ManipulationRoughVisionEnvCfg

    def __init__(self, cfg: ManipulationFlatEnvCfg | ManipulationRoughBlindEnvCfg | ManipulationRoughVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._processed_actions_with_arm = torch.zeros(self.num_envs, 18, device=self.device)

        # Reset joints smoothness
        self._previous_joints_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_previous_joints_vel = self._previous_joints_vel[:].clone()


        # pose ee commands #TODO add roll pitch yaw
        self._ee_commands = torch.zeros(self.num_envs, 7, device=self.device)

        # initial robot orientation
        self._initial_root_quat = self._robot.data.root_quat_w.clone()

        # Observation history arm
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)


        # Periodic gait
        self._step_freq = torch.tensor(self.cfg.desired_step_freq, device=self.device)
        self._duty_factor = torch.tensor(self.cfg.desired_duty_factor, device=self.device)
        self._phase_offset = torch.tensor(self.cfg.desired_phase_offset, device=self.device).repeat(self.num_envs,1)
        self._phase_signal = self._phase_offset.clone()# + self.step_dt * self._step_freq * torch.rand(self.num_envs, 1, device=self.device)*10.
        self._phase_signal = self._phase_signal % 1.0

        # Observation history locomotion
        self._observation_history_locomotion = torch.zeros(self.num_envs, cfg.locomotion_policy_env_cfg["history_length"], cfg.locomotion_policy_env_cfg["single_observation_space"], device=self.device)

        #self._locomotion_policy_onnx = ort.InferenceSession(cfg.locomotion_policy_folder_path + "/exported/policy.onnx")
        self._locomotion_policy = torch.load(
            cfg.locomotion_policy_folder_path + "/exported/policy.pt",
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=False  # <-- add this
        )
        self._locomotion_policy.eval()
        
        single_action_space_locomotion = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(cfg.action_space_locomotion,), dtype=float)
        self._actions_locomotion = torch.zeros(
            self.num_envs, gym.spaces.flatdim(single_action_space_locomotion), device=self.device
        )
        self._previous_actions_locomotion = torch.zeros(
            self.num_envs, gym.spaces.flatdim(single_action_space_locomotion), device=self.device
        )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ee_position_exp",
                "ee_final_orientation_exp",
                "ee_final_vel_exp",

                "undesired_contacts",
                "action_rate_l2",
                "action_smoothness_l2",
                "action_pose_and_vel_near_zero_l2",

                #"joints_pos_l2": joints_arm_position_reward * self.cfg.joints_arm_position_reward_scale * self.step_dt,
                "joints_vel_l2",
                "joints_final_vel_exp",
                #"joints_vel_smoothness_l2": joints_vel_smoothness * self.cfg.joints_vel_smoothness_reward_scale * self.step_dt,
                "joints_acc_l2",
                "joints_torques_l2",
                "joints_energy_l1",

                # Robot Base stability
                "base_ang_vel_l2",
                "base_lin_vel_z_l2",
                
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._arm_links_ids, _ = self._contact_sensor.find_bodies("link.*")
        self._undesired_contact_body_ids = self._base_id + self._hip_ids + self._thigh_ids + self._arm_links_ids

        
        self._feet_ids_robot, _ = self._robot .find_bodies(".*foot")
        self._hip_ids_robot, _ = self._robot.find_bodies(".*hip")
        self._ee_id_robot, _ = self._robot.find_bodies("link06")

        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]
        self._ids_only_legs_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order[0:12], preserve_order=True)[0]
        self._ids_only_arms_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order[12:18], preserve_order=True)[0]

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        self.ee_markers = VisualizationMarkers(self.cfg.frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # we add an imu
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        
        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp 
        else:
            self._processed_actions = self.cfg.action_scale * self._actions
        self._processed_actions[:,0:6] += self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order]
        
        arm_commands = self._processed_actions[:, 0:6]
        pose_commands = self._processed_actions[:, 6:8]

        if(self.cfg.use_velocity_commands == False):
            velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            velocity_commands = self._processed_actions[:, 8:]
            # Clamp velocity commands
            velocity_commands[:, 0] = torch.clamp(velocity_commands[:, 0], -0.1, 0.1)
            velocity_commands[:, 1] = torch.clamp(velocity_commands[:, 1], -0.1, 0.1)
            velocity_commands[:, 2] = torch.clamp(velocity_commands[:, 2], -0.1, 0.1)

        # Clamp ee commands
        pose_commands[:,0] = torch.clamp(pose_commands[:,0], -0.3, 0.3)
        pose_commands[:,1] = torch.clamp(pose_commands[:,1], -0.2, 0.0)
        # Get locomotion policy action
        locomotion_actions = self._get_locomotion_policy_action(pose_commands, velocity_commands)
        self._processed_actions_with_arm[:, self._ids_only_arms_joints_order] = arm_commands
        self._processed_actions_with_arm[:, self._ids_only_legs_joints_order] = locomotion_actions


    def _apply_action(self):
        # Reset robot state #TODO eliminate
        #env_ids = self._robot._ALL_INDICES
        #default_root_state = self._robot.data.default_root_state[env_ids]
        #default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        #self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        #self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._robot.set_joint_position_target(self._processed_actions_with_arm)


    def _get_observations(self) -> dict:
        
        # This is a custom event, to be moved in custom_events.py
        self._get_new_random_commands()


        # Observation --------------------------------------------------------------------------------------
        clock_data = None
        if(self.cfg.use_clock_signal):
            clock_data = torch.vstack([self._phase_signal[:,0], self._phase_signal[:,1], self._phase_signal[:,2], self._phase_signal[:,3]]).T
            # all the envs that are not moving, we put -1
            clock_data[:, :] = -1.0
            

        # Choosing the main source of observation
        if(self.cfg.use_imu):
            # Using directly the IMU
            velocity_b = self._imu.data.lin_acc_b
            angular_velocity_b = self._imu.data.ang_vel_b
            projected_gravity_b = self._imu.data.projected_gravity_b
        else:
            #Using a model-based state estimation
            velocity_b = self._robot.data.root_lin_vel_b
            angular_velocity_b = self._robot.data.root_ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b

        
        # Standard Obs for the Actor/Critic
        ROT_H2W_env = math_utils.matrix_from_quat(math_utils.yaw_quat(self._initial_root_quat))
        ee_position_commands_local_env_w = torch.matmul(ROT_H2W_env, self._ee_commands[:, :3].unsqueeze(2))
        ee_position_commands_env_w = ee_position_commands_local_env_w[:,:,0] + self._robot.data.default_root_state[:,0:3] + self.scene.env_origins
        ROT_H2W_robot = math_utils.matrix_from_quat(math_utils.yaw_quat(self._robot.data.root_quat_w))
        ee_position_commands_local_robot_w = ee_position_commands_env_w - self._robot.data.root_state_w[:, :3]
        ee_position_commands_local_robot_h = torch.matmul(ROT_H2W_robot.transpose(1,2), ee_position_commands_local_robot_w.unsqueeze(2)).squeeze(2)

        # trasform ee target orientation w in horizontal frame TODO
        # 1. Get yaw-only quaternion of the robot base
        yaw_quat = math_utils.yaw_quat(self._robot.data.root_quat_w)  # shape [N, 4]

        # 2. Invert the yaw quaternion
        yaw_quat_inv = math_utils.quat_inv(yaw_quat)  # shape [N, 4]

        # 3. Transform target orientation from world to horizontal frame
        ee_target_quat_w = self._ee_commands[:, 3:7]  # shape [N, 4]
        ee_target_quat_h = math_utils.quat_mul(yaw_quat_inv, ee_target_quat_w)  # shape [N, 4]

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    velocity_b,
                    angular_velocity_b,
                    projected_gravity_b,
                    #self._ee_commands[:, :3], #target position w
                    #self._ee_commands[:, 3:], #target orientation w
                    ee_position_commands_local_robot_h, #target position h
                    ee_target_quat_h, #target orientation h
                    self._robot.data.joint_pos[:,self._ids_joints_order] - self._robot.data.default_joint_pos[:,self._ids_joints_order],
                    self._robot.data.joint_vel[:,self._ids_joints_order],
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)


        # Add heightmap data to obs if needed
        if isinstance(self.cfg, ManipulationRoughVisionEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)      


        # Final observations dictionary
        observations = {"policy": obs}    
        # ------------------------------------------------------------------------------------------
        return observations


    def _get_rewards(self) -> torch.Tensor:

        # tracking position ee in horizontal frame
        ROT_H2W = math_utils.matrix_from_quat(math_utils.yaw_quat(self._initial_root_quat))
        ee_position_commands_local_w = torch.matmul(ROT_H2W, self._ee_commands[:, :3].unsqueeze(2))
        ee_position_commands_w = ee_position_commands_local_w[:,:,0] + self._robot.data.default_root_state[:,0:3] + self.scene.env_origins
        #next_ee_position_w = self._imu.data.pos_w[:,:3].reshape((self._imu.data.pos_w.shape[0],1,3)) + self._robot.data.body_lin_vel_w[:, self._ee_id_robot, :] * self.step_dt
        #ee_position_error = torch.sum(torch.square(ee_position_commands_w - next_ee_position_w.reshape((self._robot.data.body_pos_w.shape[0],3))), dim=1)
        ee_position_error = torch.sum(torch.square(ee_position_commands_w - self._imu.data.pos_w[:,:3]), dim=1)
        ee_position_error_mapped = torch.exp(-ee_position_error / 0.10)
        

        # tracking orientation ee near goal
        should_freeze = ee_position_error < 0.0025 #ee position error in a radius of 5cm
        curr_quat_w = self._imu.data.quat_w
        des_quat_b = self._ee_commands[:, 3:7]
        des_quat_w = quat_mul(self._robot.data.root_quat_w, des_quat_b)
        ee_orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
        ee_orientation_error_mapped = torch.exp(-ee_orientation_error / 0.10)*should_freeze


        # regulation final velocity ee near goal (reward normalized per number of dimensions)
        ee_final_velocity_error = torch.sum(torch.square(self._imu.data.lin_vel_b), dim=1)/3.
        ee_final_velocity_error_mapped = torch.exp(-ee_final_velocity_error / 0.20)*should_freeze


        # regulation final velocity joints near goal (reward normalized per number of dimensions)
        joints_arm_final_velocity_error = torch.sum(torch.square(self._robot.data.joint_vel[:,self._ids_only_arms_joints_order]), dim=1)/6.
        joints_arm_final_velocity_reward = torch.exp(-joints_arm_final_velocity_error / 2.0)*should_freeze


        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_smoothness = torch.sum(torch.square(self._actions - 2*self._previous_actions + self._previous_previous_actions), dim=1)


        # action pose near zero
        action_pose_and_vel_near_zero = torch.sum(torch.square(self._actions[:, 6:]), dim=1)


        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        

        # joint acceleration
        joints_accel = torch.sum(torch.square(self._robot.data.joint_acc[:,self._ids_only_arms_joints_order]), dim=1)


        # joint velocity
        joints_vel = torch.sum(torch.square(self._robot.data.joint_vel[:,self._ids_only_arms_joints_order]), dim=1)


        # joint torques
        joints_torques = torch.sum(torch.square(self._robot.data.applied_torque[:,self._ids_only_arms_joints_order]), dim=1)


        # energy = torque * velocity
        joints_energy = torch.sum(torch.abs(self._robot.data.applied_torque[:,self._ids_only_arms_joints_order] * self._robot.data.joint_vel[:,self._ids_only_arms_joints_order]), dim=1)


        # joints position
        joints_arm_position = self._robot.data.joint_pos[:,self._ids_only_arms_joints_order[0:4]]
        joints_arm_position_error = torch.square(joints_arm_position - self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order[0:4]])
        joints_arm_position_reward = torch.sum(joints_arm_position_error,dim=1)



        # joints vel smoothness 
        #joints_vel_smoothness = torch.sum(torch.square(self._robot.data.joint_vel[:,self._ids_only_arms_joints_order] - 2*self._previous_joints_vel + self._previous_previous_joints_vel), dim=1)
        #self._previous_previous_joints_vel = self._previous_joints_vel.clone()
        #self._previous_joints_vel = self._robot.data.joint_vel[:,self._ids_only_arms_joints_order].clone()

        # angular velocity x/y tracking
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :3]), dim=1)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])


        # Nan and Inf check
        """total_nans_check_ee_pose_error_mapped = torch.isnan(ee_pose_error_mapped * self.cfg.ee_pose_reward_scale * self.step_dt).sum()
        total_nans_check_action_rate = torch.isnan(action_rate * self.cfg.action_rate_reward_scale * self.step_dt).sum()
        total_nans_check_action_smoothness = torch.isnan(action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt).sum()
        total_nans_check_contacts = torch.isnan(contacts * self.cfg.undersired_contact_reward_scale * self.step_dt).sum()
        total_nans_check_joints_accel = torch.isnan(joints_accel * self.cfg.joints_accel_reward_scale * self.step_dt).sum()
        total_nans_check_joints_torques = torch.isnan(joints_torques * self.cfg.joints_torque_reward_scale * self.step_dt).sum()
        total_nans_check_joints_energy = torch.isnan(joints_energy * self.cfg.joints_energy_reward_scale * self.step_dt).sum()
        total_nan_check = total_nans_check_ee_pose_error_mapped + total_nans_check_action_rate + total_nans_check_action_smoothness + total_nans_check_contacts + \
                total_nans_check_joints_accel + total_nans_check_joints_torques + total_nans_check_joints_energy
        if total_nan_check > 0:
            print("Nans in reward computation")
            breakpoint()

        total_infs_check_ee_pose_error_mapped = torch.isinf(ee_pose_error_mapped * self.cfg.ee_pose_reward_scale * self.step_dt).sum()
        total_infs_check_action_rate = torch.isinf(action_rate * self.cfg.action_rate_reward_scale * self.step_dt).sum()
        total_infs_check_action_smoothness = torch.isinf(action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt).sum()
        total_infs_check_contacts = torch.isinf(contacts * self.cfg.undersired_contact_reward_scale * self.step_dt).sum()
        total_infs_check_joints_accel = torch.isinf(joints_accel * self.cfg.joints_accel_reward_scale * self.step_dt).sum()
        total_infs_check_joints_torques = torch.isinf(joints_torques * self.cfg.joints_torque_reward_scale * self.step_dt).sum()
        total_infs_check_joints_energy = torch.isinf(joints_energy * self.cfg.joints_energy_reward_scale * self.step_dt).sum()
        total_inf_check = total_infs_check_ee_pose_error_mapped + total_infs_check_action_rate + total_infs_check_action_smoothness + total_infs_check_contacts + \
                total_infs_check_joints_accel + total_infs_check_joints_torques + total_infs_check_joints_energy
        if total_inf_check > 0:
            print("Infs in reward computation")
            breakpoint()"""



        rewards = {
            # End Effector
            "ee_position_exp": ee_position_error_mapped * self.cfg.ee_position_reward_scale * self.step_dt,
            "ee_final_orientation_exp": ee_orientation_error_mapped * self.cfg.ee_final_orientation_reward_scale * self.step_dt,
            "ee_final_vel_exp": ee_final_velocity_error_mapped * self.cfg.ee_final_velocity_reward_scale * self.step_dt,

            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness_l2": action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt,
            "action_pose_and_vel_near_zero_l2": action_pose_and_vel_near_zero * self.cfg.action_pose_and_vel_near_zero_reward_scale * self.step_dt,

            #"joints_pos_l2": joints_arm_position_reward * self.cfg.joints_arm_position_reward_scale * self.step_dt,
            "joints_vel_l2": joints_vel * self.cfg.joints_vel_reward_scale * self.step_dt,
            "joints_final_vel_exp": joints_arm_final_velocity_reward * self.cfg.joints_vel_final_reward_scale * self.step_dt,
            #"joints_vel_smoothness_l2": joints_vel_smoothness * self.cfg.joints_vel_smoothness_reward_scale * self.step_dt,
            "joints_acc_l2": joints_accel * self.cfg.joints_accel_reward_scale * self.step_dt,
            "joints_torques_l2": joints_torques * self.cfg.joints_torque_reward_scale * self.step_dt,
            "joints_energy_l1": joints_energy * self.cfg.joints_energy_reward_scale * self.step_dt,

            # Robot Base stability
            "base_ang_vel_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "base_lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_check_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        died_check_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_ids], dim=-1), dim=1)[0] > 1.0, dim=1) 
        died_arms_collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._arm_links_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        died = torch.logical_or(died_check_base, died_check_hips)
        died = torch.logical_or(died, died_arms_collision)
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs: 
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0

        # Sample new commands
        self._ee_commands[env_ids, 0] = torch.zeros_like(self._ee_commands[env_ids,0]).uniform_(0.5, 0.8)
        self._ee_commands[env_ids, 1] = torch.zeros_like(self._ee_commands[env_ids,1]).uniform_(-0.3, 0.3)
        self._ee_commands[env_ids, 2] = torch.zeros_like(self._ee_commands[env_ids,2]).uniform_(-0.3, 0.0)

        desired_roll = torch.zeros_like(self._ee_commands[env_ids,3]).uniform_(-0.1, 0.1)
        desired_pitch = torch.zeros_like(self._ee_commands[env_ids,4]).uniform_(-0.1, 0.1)
        desired_yaw = torch.zeros_like(self._ee_commands[env_ids,5]).uniform_(-1., 1.)
        self._ee_commands[env_ids,3:] = math_utils.quat_from_euler_xyz(desired_roll, desired_pitch, desired_yaw)

        # Reset contact periodic
        self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()# + self.step_dt * self._step_freq * torch.rand(env_ids.shape[0], 1, device=self.device)*10.
        self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0


        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        
        #joint_pos[:, self._ids_only_arms_joints_order] += torch.zeros_like(joint_pos[:, self._ids_only_arms_joints_order]).uniform_(-3.14, 3.14)
        # we need to project them inside the robots limits - natural constraint!
        #joints_limits = self._robot.data.default_joint_pos_limits
        #joints_arm_limits = joints_limits[:,self._ids_only_arms_joints_order]
        #joint_pos[:, self._ids_only_arms_joints_order] = torch.clamp(joint_pos[:, self._ids_only_arms_joints_order], joints_arm_limits[0,:,0], joints_arm_limits[0,:,1])

        # we need to project them inside the robots limits - smaller constraint!
        joints_arm_limits_start_upper = torch.tensor([1.0, 2.6, -0.5, 1.5184, 1.3439, 2.7925], device=self.device) 
        joints_arm_limits_start_lower = torch.tensor([-1.0, 1.6, -1.8, -1.5184, -1.3439, -2.7925], device=self.device) 
        random_tensor = torch.rand((env_ids.shape[0],6), device=self.device)
        sampled_arm_joints = joints_arm_limits_start_lower + (joints_arm_limits_start_upper - joints_arm_limits_start_lower) * random_tensor
        joint_pos[:, self._ids_only_arms_joints_order] = sampled_arm_joints

        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        #default_root_state[:, 3:7] = math_utils.random_yaw_orientation(env_ids.shape[0], device=self.device)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset joints smoothness
        self._previous_joints_vel[env_ids] = joint_vel[:, self._ids_only_arms_joints_order].clone()*0.0
        self._previous_previous_joints_vel[env_ids] = self._previous_joints_vel[env_ids].clone()

        # Save initial orientation
        self._initial_root_quat[env_ids] = default_root_state[:, 3:7].clone()
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())
        
        self.extras["log"].update(extras)



    def _get_new_random_commands(self):
        #distance_between_ee_and_command = torch.norm(self._imu.data.pos_w[:,0:3] - self._ee_commands[:,0:3], dim=1)
        #random_sample_scalar = torch.rand((self._robot._ALL_INDICES, 6), device=self.device)

        # Sample new commands
        commands_resample = torch.zeros_like(self._ee_commands)
        commands_resample[:, 0] = torch.zeros_like(self._ee_commands[:, 0]).uniform_(0.5, 0.8)
        commands_resample[:, 1] = torch.zeros_like(self._ee_commands[:, 1]).uniform_(-0.3, 0.3)
        commands_resample[:, 2] = torch.zeros_like(self._ee_commands[:, 2]).uniform_(-0.3, 0.0)
        desired_roll = torch.zeros_like(self._ee_commands[:,3]).uniform_(-0.1, 0.1)
        desired_pitch = torch.zeros_like(self._ee_commands[:,4]).uniform_(-0.1, 0.1)
        #desired_pitch = torch.zeros_like(self._ee_commands[:,4]).uniform_(1.4, 1.5)
        desired_yaw = torch.zeros_like(self._ee_commands[:,5]).uniform_(-1., 1.)
        commands_resample[:, 3:] = math_utils.quat_from_euler_xyz(desired_roll, desired_pitch, desired_yaw)

        resample_time = self.episode_length_buf == self.max_episode_length - 250
        self._ee_commands = self._ee_commands * ~resample_time.unsqueeze(1).expand(-1, 7) + commands_resample * resample_time.unsqueeze(1).expand(-1, 7)

        resample_time_2 = self.episode_length_buf == self.max_episode_length - 500
        self._ee_commands = self._ee_commands * ~resample_time_2.unsqueeze(1).expand(-1, 7) + commands_resample * resample_time_2.unsqueeze(1).expand(-1, 7)

        resample_time_3 = self.episode_length_buf == self.max_episode_length - 750
        self._ee_commands = self._ee_commands * ~resample_time_3.unsqueeze(1).expand(-1, 7) + commands_resample * resample_time_3.unsqueeze(1).expand(-1, 7)

        # visualize goal
        ROT_H2W = math_utils.matrix_from_quat(math_utils.yaw_quat(self._initial_root_quat))  
        ee_position_commands_local_w = torch.matmul(ROT_H2W, self._ee_commands[:, :3].unsqueeze(2))
        goal_pos = ee_position_commands_local_w[:,:,0] + self._robot.data.default_root_state[:,0:3] + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self._ee_commands[:, 3:7])
        
        # visualize current ee
        self.ee_markers.visualize(self._imu.data.pos_w[:,0:3], self._imu.data.quat_w)



    def _get_locomotion_policy_action(self, pose_commands, velocity_commands):
        # Observation --------------------------------------------------------------------------------------
        clock_data = None
        if(self.cfg.use_clock_signal):
            clock_data = torch.vstack([self._phase_signal[:,0], self._phase_signal[:,1], self._phase_signal[:,2], self._phase_signal[:,3]]).T
            # all the envs that are not moving, we put -1
            should_move = torch.norm(velocity_commands[:, :3], dim=1) > 0.01
            clock_data[:, :] = clock_data[:, :]*should_move.unsqueeze(1).expand(-1, 4) + -1.0* ~should_move.unsqueeze(1).expand(-1, 4)
            

        # Choosing the main source of observation
        if(self.cfg.use_imu):
            # Using directly the IMU
            velocity_b = self._imu.data.lin_acc_b
            angular_velocity_b = self._imu.data.ang_vel_b
            projected_gravity_b = self._imu.data.projected_gravity_b
        else:
            #Using a model-based state estimation
            velocity_b = self._robot.data.root_lin_vel_b
            angular_velocity_b = self._robot.data.root_ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b



        # Standard Obs for the Actor/Critic
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    velocity_b,
                    angular_velocity_b,
                    projected_gravity_b,
                    velocity_commands,
                    pose_commands,
                    self._robot.data.joint_pos[:,self._ids_only_legs_joints_order] - self._robot.data.default_joint_pos[:,self._ids_only_legs_joints_order],
                    self._robot.data.joint_vel[:,self._ids_only_legs_joints_order],
                    self._actions_locomotion,
                    clock_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history_locomotion):
            #the bottom element is the newest observation!!
            self._observation_history_locomotion = torch.cat((self._observation_history_locomotion[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history_locomotion, start_dim=1)

        # Add joint arm info
        joints_arm = self._robot.data.joint_pos[:,self._ids_only_arms_joints_order] - self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order]
        obs = torch.cat((obs, joints_arm), dim=-1)


        with torch.no_grad():
            self._actions_locomotion = self._locomotion_policy(obs)
        # For check results against ONNX
        #obs_onnx = obs.detach().cpu().numpy()
        #self._actions_locomotion = self._locomotion_policy_onnx.run(None, {'obs': obs_onnx[1].reshape((1,330))})

        # Clip the action
        self._actions_locomotion = torch.clamp(self._actions_locomotion, -self.cfg.desired_clip_actions_locomotion, self.cfg.desired_clip_actions_locomotion)

        # Filter the action
        if(self.cfg.use_filter_actions_locomotion):
            alpha = 0.8
            temp = alpha * self._actions_locomotion + (1 - alpha) * self._previous_actions_locomotion
            self._processed_actions_locomotion = self.cfg.action_scale_locomotion * temp + self._robot.data.default_joint_pos[:,self._ids_only_legs_joints_order]
        else:
            self._processed_actions_locomotion = self.cfg.action_scale_locomotion * self._actions_locomotion + self._robot.data.default_joint_pos[:,self._ids_only_legs_joints_order]

        return self._processed_actions_locomotion


