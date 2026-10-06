# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv



def randomize_joint_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if env_ids != slice(None) and joint_ids != slice(None):
        env_ids_for_slice = env_ids[:, None]
    else:
        env_ids_for_slice = env_ids

    # sample joint properties from the given ranges and set into the physics simulation
    # joint friction coefficient
    if friction_distribution_params is not None:
        friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )

        # ensure the friction coefficient is non-negative
        friction_coeff = torch.clamp(friction_coeff, min=0.0)

        # Always set static friction (indexed once)
        static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

        # Randomize raw tensors
        #dynamic_friction_coeff = _randomize_prop_by_op(
        #    asset.data.default_joint_dynamic_friction_coeff.clone(),
        #    friction_distribution_params,
        #    env_ids,
        #    joint_ids,
        #    operation=operation,
        #    distribution=distribution,
        #)
        viscous_friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_viscous_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )

        # Clamp to non-negative
        #dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
        viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

        # Ensure dynamic ≤ static (same shape before indexing)
        #dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

        # Index once at the end
        #dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]
        viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]


        # Newton exposes a single dry-friction value, while PhysX backends also
        # provide a separate dynamic-friction coefficient.
        if hasattr(asset, "write_joint_dynamic_friction_coefficient_to_sim_index"):
            asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=static_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        else:
            asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

    # joint armature
    if armature_distribution_params is not None:
        armature = _randomize_prop_by_op(
            asset.data.default_joint_armature.clone(),
            armature_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_armature_to_sim_index(
            armature=armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
        )


def randomize_joint_friction_model(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    first_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    second_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the friction parameters used in joint friction model. 
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if friction_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                friction = actuator.friction_static.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_static[env_ids[:, None], actuator_joint_ids] = friction

                friction = actuator.friction_dynamic.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_dynamic[env_ids[:, None], actuator_joint_ids] = friction

    if armature_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                armature = actuator.armature.to(asset.device).clone()
                armature = _randomize_prop_by_op(
                    armature, armature_distribution_params, env_ids, torch.arange(armature.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.armature[env_ids[:, None], actuator_joint_ids] = armature


def randomize_joint_delay_model(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    first_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    second_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):

    """
    Randomize the delay used in joint hydraulic model. 
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if first_order_delay_filter_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                first_order_delay_filter = actuator.first_order_delay_filter.to(asset.device).clone()
                first_order_delay_filter = _randomize_prop_by_op(
                    first_order_delay_filter, first_order_delay_filter_distribution_params, env_ids, torch.arange(first_order_delay_filter.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.first_order_delay_filter[env_ids[:, None], actuator_joint_ids] = first_order_delay_filter

    if second_order_delay_filter_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                second_order_delay_filter = actuator.second_order_delay_filter.to(asset.device).clone()
                second_order_delay_filter = _randomize_prop_by_op(
                    second_order_delay_filter, second_order_delay_filter_distribution_params, env_ids, torch.arange(second_order_delay_filter.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.second_order_delay_filter[env_ids[:, None], actuator_joint_ids] = second_order_delay_filter


def randomize_pace_actuator_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    min_delay: int = 0,
    max_delay: int | None = None,
):
    """Randomize integer delay (in simulation steps) for Pace actuators.

    The sampled delay is applied per-environment at reset through each actuator's
    ``update_time_lags`` method.
    """
    if min_delay < 0:
        raise ValueError(f"min_delay must be >= 0, got {min_delay}.")

    asset: Articulation = env.scene[asset_cfg.name]

    # If max_delay is not specified in the event params, use the maximum default
    # max_delay declared in the Pace actuator configs attached to this asset.
    if max_delay is None:
        inferred_max_delay = 0
        for actuator in asset.actuators.values():
            actuator_cfg = getattr(actuator, "cfg", None)
            actuator_max_delay = getattr(actuator_cfg, "max_delay", None)
            if actuator_max_delay is not None:
                inferred_max_delay = max(inferred_max_delay, int(actuator_max_delay))
        max_delay = inferred_max_delay

    if max_delay < min_delay:
        raise ValueError(f"max_delay must be >= min_delay, got [{min_delay}, {max_delay}].")

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if len(env_ids) == 0:
        return

    delays = torch.randint(
        low=min_delay,
        high=max_delay + 1,
        size=(len(env_ids),),
        dtype=torch.int,
        device=asset.device,
    )

    for actuator in asset.actuators.values():
        if hasattr(actuator, "update_time_lags"):
            actuator.update_time_lags(delays, env_ids)


def _sample_arm_trajectory(self, env_ids: torch.Tensor):
    """Sample a new arm target and trajectory timing for the selected environments."""
    num_envs = env_ids.shape[0]
    self._arm_trajectory_start_pos[env_ids] = self._joints_arm_command_pos[env_ids]

    arm_limits = self._robot.data.default_joint_pos_limits[env_ids][:, self._ids_only_arms_joints_order]
    lower_limits = arm_limits[..., 0]
    upper_limits = arm_limits[..., 1]
    self._arm_trajectory_target_pos[env_ids] = lower_limits + torch.rand_like(lower_limits) * (
        upper_limits - lower_limits
    )

    duration_min, duration_max = self.cfg.arm_trajectory_duration_range_s
    self._arm_trajectory_duration_s[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
        duration_min, duration_max
    )

    update_min, update_max = self.cfg.arm_target_update_interval_range_s
    self._arm_target_update_interval_s[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
        update_min, update_max
    )
    self._arm_trajectory_elapsed_s[env_ids] = 0.0
    self._arm_target_elapsed_s[env_ids] = 0.0


def _update_arm_trajectory(self):
    """Advance every arm along its current linear joint-space trajectory."""
    num_episode_from_start = self.common_step_counter / 24. #self.max_episode_length #HACK this should be taken from rsl rl
    if(num_episode_from_start >= self.cfg.arm_trajectory_event_start):
        self._arm_trajectory_elapsed_s += self.step_dt
        self._arm_target_elapsed_s += self.step_dt

        interpolation = torch.clamp(
            self._arm_trajectory_elapsed_s / self._arm_trajectory_duration_s,
            min=0.0,
            max=1.0,
        ).unsqueeze(1)
        self._joints_arm_command_pos = torch.lerp(
            self._arm_trajectory_start_pos,
            self._arm_trajectory_target_pos,
            interpolation,
        )

        resample_env_ids = torch.nonzero(
            self._arm_target_elapsed_s >= self._arm_target_update_interval_s,
            as_tuple=False,
        ).squeeze(-1)
        if resample_env_ids.numel() > 0:
            _sample_arm_trajectory(self, resample_env_ids)


def _sample_random_commands(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    num_commands = self.num_envs if env_ids is None else env_ids.shape[0]
    commands = torch.empty(num_commands, self._velocity_commands.shape[1], device=self.device, dtype=self._velocity_commands.dtype)
    commands.uniform_(-1.0, 1.0)
    commands[:, 0] *= 0.5
    commands[:, 1] *= 0.25
    commands[:, 2] *= 0.5
    return commands


def _get_new_random_commands(self, env_ids: torch.Tensor | None = None):
    if env_ids is not None:
        self._velocity_commands[env_ids, :3] = _sample_random_commands(self, env_ids)

    resample_time = self.episode_length_buf == self.max_episode_length - 300
    commands_resample = torch.zeros_like(self._velocity_commands).uniform_(-1.0, 1.0)
    commands_resample = _sample_random_commands(self)
    self._velocity_commands[:, :3] = self._velocity_commands[:, :3] * ~resample_time.unsqueeze(1).expand(-1, 3) + commands_resample * resample_time.unsqueeze(1).expand(-1, 3)

    # Stop and small pose commands
    rest_time = self.episode_length_buf >= self.max_episode_length - 150
    specific_rest_time = self.episode_length_buf == self.max_episode_length - 100
    self._velocity_commands[:, :3] *= ~rest_time.unsqueeze(1).expand(-1, 3)
    self._pose_commands[:, 0] = self._pose_commands[:, 0] * ~specific_rest_time + torch.zeros_like(self._pose_commands[:,0]).uniform_(-0.3, 0.3) * specific_rest_time
    self._pose_commands[:, 1] = self._pose_commands[:, 1] * ~specific_rest_time + torch.zeros_like(self._pose_commands[:,1]).uniform_(-0.1, 0.05) * specific_rest_time
    

    # Took some envs, and put to zero the vel
    num_fixed_envs = 500
    if self.num_envs > num_fixed_envs:
        fixed_env_ids = torch.arange(num_fixed_envs, device=self.device)
        self._velocity_commands[fixed_env_ids, :3] *= 0.0
