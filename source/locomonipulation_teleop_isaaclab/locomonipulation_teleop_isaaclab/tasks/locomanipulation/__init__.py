# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
from .locomotion_env import LocomotionEnv


# Aliengo environments
from .locomotion_env import AliengoFlatEnvCfg, AliengoRoughVisionEnvCfg, AliengoRoughBlindEnvCfg

gym.register(
    id="Locomotion-Aliengo-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

from .manipulation_env import ManipulationEnv
from .manipulation_env import ManipulationFlatEnvCfg, ManipulationRoughBlindEnvCfg, ManipulationRoughVisionEnvCfg

gym.register(
    id="Manipulation-Aliengo-Flat",
    entry_point=ManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ManipulationFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Manipulation-Aliengo-Rough-Blind",
    entry_point=ManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ManipulationRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Manipulation-Aliengo-Rough-Vision",
    entry_point=ManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ManipulationRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

