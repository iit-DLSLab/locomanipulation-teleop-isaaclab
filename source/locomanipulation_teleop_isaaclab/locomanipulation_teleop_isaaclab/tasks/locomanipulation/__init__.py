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
from .locomanipulation_env import LocomotionManipulationEnv


# Go2 + Piper environments
from .locomanipulation_env import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg

gym.register(
    id="LocoManipulation-Go2-Flat",
    entry_point=LocomotionManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="LocoManipulation-Go2-Rough-Blind",
    entry_point=LocomotionManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="LocoManipulation-Go2-Rough-Vision",
    entry_point=LocomotionManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)


# A2 + Piper environments
from .locomanipulation_env import A2FlatEnvCfg, A2RoughVisionEnvCfg, A2RoughBlindEnvCfg

gym.register(
    id="LocoManipulation-A2-Flat",
    entry_point=LocomotionManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": A2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="LocoManipulation-A2-Rough-Blind",
    entry_point=LocomotionManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": A2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="LocoManipulation-A2-Rough-Vision",
    entry_point=LocomotionManipulationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": A2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)
