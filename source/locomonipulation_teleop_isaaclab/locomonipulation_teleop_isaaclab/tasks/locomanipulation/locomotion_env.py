# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

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


from .locomotion_env_cfg import AliengoFlatEnvCfg, AliengoRoughBlindEnvCfg, AliengoRoughVisionEnvCfg

class LocomotionEnv(DirectRLEnv):
    cfg: AliengoFlatEnvCfg | AliengoRoughBlindEnvCfg | AliengoRoughVisionEnvCfg 

    def __init__(self, cfg: AliengoFlatEnvCfg | AliengoRoughBlindEnvCfg | AliengoRoughVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._pose_commands = torch.zeros(self.num_envs, 2, device=self.device) # pitch height

        # Arm fixed position
        self._joints_arm_fixed_pos = torch.zeros(self.num_envs, 6, device=self.device)

        # Swing peak
        self._swing_peak = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        
        # Desired Hip Offset
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        
        # Periodic gait
        self._step_freq = torch.tensor(self.cfg.desired_step_freq, device=self.device)
        self._duty_factor = torch.tensor(self.cfg.desired_duty_factor, device=self.device)
        self._phase_offset = torch.tensor(self.cfg.desired_phase_offset, device=self.device).repeat(self.num_envs,1)
        self._phase_signal = self._phase_offset.clone()# + self.step_dt * self._step_freq * torch.rand(self.num_envs, 1, device=self.device)*10.
        self._phase_signal = self._phase_signal % 1.0


        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)


        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_height_exp",
                "track_lin_vel_xy_exp",
                "track_lin_vel_z_l2",
                "track_orientation_l2",
                "track_ang_vel_xy_l2",
                "track_ang_vel_z_exp",

                "undesired_contacts",
                "action_rate_l2",
                "action_smoothness_l2",
                
                "joints_hip_pos_l2",
                "joints_thigh_pos_l2",
                "joints_calf_pos_l2",
                "joints_acc_l2",
                "joints_torques_l2",
                "joints_energy_l1",
                
                "feet_height_clearance",
                "feet_contact_suggestion",
                "feet_to_hip_distance_l2",
                "feet_vertical_surface_contacts",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._undesired_contact_body_ids = self._base_id + self._hip_ids + self._thigh_ids

        
        self._feet_ids_robot, _ = self._robot .find_bodies(".*foot")
        self._hip_ids_robot, _ = self._robot.find_bodies(".*hip")
        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]
        self._ids_only_legs_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order[0:12], preserve_order=True)[0]
        self._ids_only_arms_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order[12:18], preserve_order=True)[0]


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
            #self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos[:,0:12]
            self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos[:,self._ids_joints_order[0:12]]
        else:
            #self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:,0:12]
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:,self._ids_joints_order[0:12]]


    def _apply_action(self):
        processed_actions_with_arm = torch.zeros(self.num_envs, 18, device=self.device)
        processed_actions_with_arm[:, self._ids_only_arms_joints_order] = self._joints_arm_fixed_pos.clone()
        processed_actions_with_arm[:, self._ids_only_legs_joints_order] = self._processed_actions
        self._robot.set_joint_position_target(processed_actions_with_arm)


    def _get_observations(self) -> dict:
        
        # This is a custom event, to be moved in custom_events.py
        self._get_new_random_commands()


        # Observation --------------------------------------------------------------------------------------
        clock_data = None
        if(self.cfg.use_clock_signal):
            clock_data = torch.vstack([self._phase_signal[:,0], self._phase_signal[:,1], self._phase_signal[:,2], self._phase_signal[:,3]]).T
            # all the envs that are not moving, we put -1
            should_move = torch.norm(self._velocity_commands[:, :3], dim=1) > 0.01
            clock_data[:, :] = clock_data[:, :]*should_move.unsqueeze(1).expand(-1, 4) + -1.0* ~should_move.unsqueeze(1).expand(-1, 4)
            

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
                    self._velocity_commands,
                    self._pose_commands,
                    self._robot.data.joint_pos[:,self._ids_only_legs_joints_order] - self._robot.data.default_joint_pos[:,self._ids_only_legs_joints_order],
                    self._robot.data.joint_vel[:,self._ids_only_legs_joints_order],
                    self._actions,
                    clock_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)
        
        # Add joint arm info
        joints_arm = self._robot.data.joint_pos[:,self._ids_only_arms_joints_order] - self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order]
        obs = torch.cat((obs, joints_arm), dim=-1)


        # Add heightmap data to obs if needed
        if isinstance(self.cfg, AliengoRoughVisionEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)      


        # Final observations dictionary
        observations = {"policy": obs}    
        

        # Critic OBS could be different if needed
        if(self.cfg.use_asymmetric_ppo):
            obs_critic = self._get_privileged_observation()
            observations["critic"] = torch.cat((obs, obs_critic), dim=-1)
        # ------------------------------------------------------------------------------------------

        return observations


    def _get_rewards(self) -> torch.Tensor:

        # track_height
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)

        height_error = torch.square(self.cfg.desired_base_height + mean_height_ray + self._pose_commands[:,1] - self._robot.data.root_state_w[:, 2])
        height_error_mapped = torch.exp(-height_error / 0.01)


        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._velocity_commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        

        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])


        # terrain orientation
        height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
        height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
        height_map_y_points = int(round(self._height_scanner.cfg.pattern_cfg.size[1] / height_map_resolution))
        distance_between_front_and_back = (height_map_x_points/2)* height_map_resolution

        cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_back = cols_back.flatten().to(height_data_scanner.device)
        selected_height_data_back = height_data_scanner[:, cols_back]

        cols_front = torch.arange(int(height_map_x_points/2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_front = cols_front.flatten().to(height_data_scanner.device)
        selected_height_data_front = height_data_scanner[:, cols_front]

        mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
        mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
        delta_z = mean_height_ray_front - mean_height_ray_back
        delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
        terrain_pitch = -torch.atan2(delta_z, delta_s)
        #terrain_pitch = torch.atan2(torch.sin(terrain_pitch), torch.cos(terrain_pitch))
        
        root_roll_w, root_pitch_w, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        root_roll_w = torch.atan2(torch.sin(root_roll_w), torch.cos(root_roll_w))
        root_pitch_w = torch.atan2(torch.sin(root_pitch_w), torch.cos(root_pitch_w))
        
        #base_orientation =  torch.square(terrain_pitch - root_pitch_w)# + torch.square(0 - root_roll_w)
        base_orientation =  torch.square(terrain_pitch + self._pose_commands[:,0] - root_pitch_w) + torch.square(0 + root_roll_w)


        # angular velocity x/y tracking
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)


        # yaw rate tracking
        yaw_rate_error = torch.square(self._velocity_commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_smoothness = torch.sum(torch.square(self._actions - 2*self._previous_actions + self._previous_previous_actions), dim=1)
        
        
        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        

        # joint acceleration
        joints_accel = torch.sum(torch.square(self._robot.data.joint_acc[:,self._ids_joints_order[0:12]]), dim=1)


        # joint torques
        joints_torques = torch.sum(torch.square(self._robot.data.applied_torque[:,self._ids_joints_order[0:12]]), dim=1)


        # energy = torque * velocity
        joints_energy = torch.sum(torch.abs(self._robot.data.applied_torque[:,self._ids_joints_order[0:12]] * self._robot.data.joint_vel[:,self._ids_joints_order[0:12]]), dim=1)

        
        # hip position
        hip_joints_position = self._robot.data.joint_pos[:,self._ids_joints_order[0:4]]
        hip_joints_position_error = torch.square(hip_joints_position - self._robot.data.default_joint_pos[:,self._ids_joints_order[0:4]])
        hip_joints_position_reward = torch.sum(hip_joints_position_error,dim=1)


        # thigh position
        thigh_joints_position = self._robot.data.joint_pos[:,self._ids_joints_order[4:8]]
        thigh_joints_position_error = torch.square(thigh_joints_position - self._robot.data.default_joint_pos[:,self._ids_joints_order[4:8]])
        thigh_joints_position_reward = torch.sum(thigh_joints_position_error,dim=1)


        # calf position
        calf_joints_position = self._robot.data.joint_pos[:,self._ids_joints_order[8:12]]
        calf_joints_position_error = torch.square(calf_joints_position - self._robot.data.default_joint_pos[:,self._ids_joints_order[8:12]])
        calf_joints_position_reward = torch.sum(calf_joints_position_error,dim=1)


        # feet periodical contacts suggestion
        should_move = torch.norm(self._velocity_commands[:, :3], dim=1) > 0.01
        self._phase_signal += self.step_dt * self._step_freq
        self._phase_signal = self._phase_signal % 1.0
        contact_periodic_on = self._phase_signal < self._duty_factor
        contacts_foot = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        feet_contact_suggestion = (torch.sum(contact_periodic_on*contacts_foot, dim=1) + \
                                   torch.sum(~contact_periodic_on*~contacts_foot, dim=1))*should_move/4.0
        feet_contact_suggestion += (torch.sum(contacts_foot, dim=1)*~should_move/4.0)
        


        # feet height clearance periodic
        feet_z_target_error = self.cfg.desired_feet_height + torch.cat((mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1) - self._robot.data.body_pos_w[:, self._feet_ids_robot, 2]
        # If the raw error is negative, halve it to not discourage too much
        feet_z_target_error = torch.where(feet_z_target_error < 0.0, feet_z_target_error * 0.2, feet_z_target_error)
        feet_z_target_error = torch.abs(feet_z_target_error)
        feet_z_target_error = torch.clamp(feet_z_target_error, min=.0, max=self.cfg.desired_feet_height)
 
        feet_height_clearance_periodic_FL = torch.exp(-feet_z_target_error[:,0]/ 0.01) * should_move * ~contact_periodic_on[:,0]
        feet_height_clearance_periodic_FR = torch.exp(-feet_z_target_error[:,1]/ 0.01) * should_move * ~contact_periodic_on[:,1]
        feet_height_clearance_periodic_RL = torch.exp(-feet_z_target_error[:,2]/ 0.01) * should_move * ~contact_periodic_on[:,2]
        feet_height_clearance_periodic_RR = torch.exp(-feet_z_target_error[:,3]/ 0.01) * should_move * ~contact_periodic_on[:,3]
        feet_height_clearance = feet_height_clearance_periodic_FL + feet_height_clearance_periodic_FR
        feet_height_clearance += feet_height_clearance_periodic_RL + feet_height_clearance_periodic_RR


        # feet to hip distance
        ROT_W2H = math_utils.matrix_from_quat(math_utils.yaw_quat(self._robot.data.root_quat_w))
        feet_to_base_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        feet_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), feet_to_base_w.transpose(1, 2))
        
        hip_to_base_w = self._robot.data.body_pos_w[:, self._hip_ids_robot, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        hip_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), hip_to_base_w.transpose(1, 2))
        
        desired_hip_offset = self._desired_hip_offset
        feet_to_hip_distance_x = torch.square(feet_to_base_h[:, 0] - hip_to_base_h[:, 0])
        feet_to_hip_distance_y = torch.square(feet_to_base_h[:, 1] + desired_hip_offset.unsqueeze(0) - hip_to_base_h[:, 1])
        feet_to_hip_distance = -torch.mean(torch.sqrt(feet_to_hip_distance_x + feet_to_hip_distance_y), dim=1)
        

        # Penalize feet hitting vertical surfaces  
        forces_z = torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2])
        forces_xy = torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids, :2], dim=2)
        feet_vertical_surface_contacts = torch.any(forces_xy > 4 * forces_z, dim=1).float()
        feet_vertical_surface_contacts *= torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


        rewards = {
            "track_height_exp": height_error_mapped * self.cfg.height_reward_scale * self.step_dt,
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "track_orientation_l2": base_orientation * self.cfg.orientation_reward_scale * self.step_dt,
            "track_ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,

            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness_l2": action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt,

            "joints_hip_pos_l2": hip_joints_position_reward * self.cfg.joints_hip_position_reward_scale * self.step_dt,
            "joints_thigh_pos_l2": thigh_joints_position_reward * self.cfg.joints_thigh_position_reward_scale * self.step_dt,
            "joints_calf_pos_l2": calf_joints_position_reward * self.cfg.joints_calf_position_reward_scale * self.step_dt,
            "joints_acc_l2": joints_accel * self.cfg.joints_accel_reward_scale * self.step_dt,
            "joints_torques_l2": joints_torques * self.cfg.joints_torque_reward_scale * self.step_dt,
            "joints_energy_l1": joints_energy * self.cfg.joints_energy_reward_scale * self.step_dt,

            "feet_height_clearance": feet_height_clearance * self.cfg.feet_height_clearance_reward_scale * self.step_dt,
            "feet_contact_suggestion": feet_contact_suggestion * self.cfg.feet_contact_suggestion_reward_scale * self.step_dt,
            "feet_to_hip_distance_l2": feet_to_hip_distance * self.cfg.feet_to_hip_distance_reward_scale * self.step_dt,
            "feet_vertical_surface_contacts": feet_vertical_surface_contacts * self.cfg.feet_vertical_surface_contacts_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Check for NaNs and Infs
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print("NaN or Inf detected in reward computation. Setting reward to zero for affected environments.")
            breakpoint()  # For debugging purposes
            reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_check_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        died_check_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_ids], dim=-1), dim=1)[0] > 1.0, dim=1) 
        died = torch.logical_or(died_check_base, died_check_hips)
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
        self._velocity_commands[env_ids] = torch.zeros_like(self._velocity_commands[env_ids]).uniform_(-1.0, 1.0)
        self._velocity_commands[env_ids, 0] *= 0.5
        self._velocity_commands[env_ids, 1] *= 0.25 
        self._velocity_commands[env_ids, 2] *= 0.3 
        self._pose_commands[env_ids, 0] = torch.zeros_like(self._pose_commands[env_ids,0]).uniform_(-0.3, 0.3)*0.0
        self._pose_commands[env_ids, 1] = torch.zeros_like(self._pose_commands[env_ids,1]).uniform_(-0.1, 0.0)*0.0

        # Reset swing peak
        self._swing_peak[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        
        # Reset contact periodic
        self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()# + self.step_dt * self._step_freq * torch.rand(env_ids.shape[0], 1, device=self.device)*10.
        self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0


        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._ids_only_arms_joints_order] += torch.zeros_like(joint_pos[:, self._ids_only_arms_joints_order]).uniform_(-3.14, 3.14)
        joint_pos[:, self._ids_only_legs_joints_order] += torch.zeros_like(joint_pos[:, self._ids_only_legs_joints_order]).uniform_(-0.2, 0.2)
        # we need to project them inside the robots limits (only arms!)
        joints_limits = self._robot.data.default_joint_pos_limits
        joints_arm_limits = joints_limits[:,self._ids_only_arms_joints_order]
        joint_pos[:, self._ids_only_arms_joints_order] = torch.clamp(joint_pos[:, self._ids_only_arms_joints_order], joints_arm_limits[0,:,0], joints_arm_limits[0,:,1])

        # we save the arm positions to keep them fixed during the episode via PD control
        self._joints_arm_fixed_pos[env_ids] = joint_pos[:, self._ids_only_arms_joints_order].clone()
        
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = math_utils.random_yaw_orientation(env_ids.shape[0], device=self.device)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
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
        resample_time = self.episode_length_buf == self.max_episode_length - 300
        commands_resample = torch.zeros_like(self._velocity_commands).uniform_(-1.0, 1.0)
        commands_resample[:, 0] *= 0.5
        commands_resample[:, 1] *= 0.25 
        commands_resample[:, 2] *= 0.3 
        self._velocity_commands[:, :3] = self._velocity_commands[:, :3] * ~resample_time.unsqueeze(1).expand(-1, 3) + commands_resample * resample_time.unsqueeze(1).expand(-1, 3)

        # Stop and small pose commands
        rest_time = self.episode_length_buf >= self.max_episode_length - 150
        specific_rest_time = self.episode_length_buf == self.max_episode_length - 100
        self._velocity_commands[:, :3] *= ~rest_time.unsqueeze(1).expand(-1, 3)
        self._pose_commands[:, 0] = self._pose_commands[:, 0] * ~specific_rest_time + torch.zeros_like(self._pose_commands[:,0]).uniform_(-0.3, 0.3) * specific_rest_time
        self._pose_commands[:, 1] = self._pose_commands[:, 1] * ~specific_rest_time + torch.zeros_like(self._pose_commands[:,1]).uniform_(-0.1, 0.0) * specific_rest_time
        

        # Took some envs, and put to zero the vel
        num_fixed_envs = 500
        if self.num_envs > num_fixed_envs:
            fixed_env_ids = torch.arange(num_fixed_envs, device=self.device)
            self._velocity_commands[fixed_env_ids, :3] *= 0.0


    def _get_privileged_observation(self):
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*"])
        asset: Articulation = self.scene[asset_cfg.name]
        
        """hip_static_friction = asset.actuators["hip"].friction_static
        thigh_static_friction = asset.actuators["thigh"].friction_static
        calf_static_friction = asset.actuators["calf"].friction_static
        
        hip_dynamic_friction = asset.actuators["hip"].friction_dynamic
        thigh_dynamic_friction = asset.actuators["thigh"].friction_dynamic
        calf_dynamic_friction = asset.actuators["calf"].friction_dynamic

        hip_armature = asset.actuators["hip"].armature
        thigh_armature = asset.actuators["thigh"].armature
        calf_armature = asset.actuators["calf"].armature

        hip_stiffness = asset.actuators["hip"].stiffness
        thigh_stiffness = asset.actuators["thigh"].stiffness
        calf_stiffness = asset.actuators["calf"].stiffness

        hip_damping = asset.actuators["hip"].damping
        thigh_damping = asset.actuators["thigh"].damping
        calf_damping = asset.actuators["calf"].damping

        #asset_cfg_base = SceneEntityCfg("robot", body_names="base")
        #asset_base = self.scene[asset_cfg_base.name]
        #masses = asset_base.root_physx_view.get_masses()
        #inertias = asset_base.root_physx_view.get_inertias()

        default_stiffness = asset.data.default_joint_stiffness[0][0]
        default_damping = asset.data.default_joint_damping[0][0]"""


        # height error
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)
        height_error = torch.abs(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])


        # terrain orientation
        height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
        height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
        height_map_y_points = int(round(self._height_scanner.cfg.pattern_cfg.size[1] / height_map_resolution))
        distance_between_front_and_back = (height_map_x_points/2)* height_map_resolution

        cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_back = cols_back.flatten().to(height_data_scanner.device)
        selected_height_data_back = height_data_scanner[:, cols_back]

        cols_front = torch.arange(int(height_map_x_points/2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_front = cols_front.flatten().to(height_data_scanner.device)
        selected_height_data_front = height_data_scanner[:, cols_front]

        mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
        mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
        delta_z = mean_height_ray_front - mean_height_ray_back
        delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
        terrain_pitch = -torch.atan2(delta_z, delta_s)

        contacts_foot = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

        obs_privileged = torch.cat(( 
                            #hip_stiffness/default_stiffness, thigh_stiffness/default_stiffness, calf_stiffness/default_stiffness, #P gain
                            #hip_damping/default_damping, thigh_damping/default_damping, calf_damping/default_damping, #D gain
                            self._robot.data.root_lin_vel_b,
                            height_error.unsqueeze(1),
                            terrain_pitch.unsqueeze(1),
                            contacts_foot,
                            #masses, inertias,
                            #hip_static_friction, thigh_static_friction, calf_static_friction,  
                            #hip_dynamic_friction, thigh_dynamic_friction, calf_dynamic_friction, 
                            #hip_armature, thigh_armature, calf_armature
                            ) 
                        , dim=-1)
        return obs_privileged