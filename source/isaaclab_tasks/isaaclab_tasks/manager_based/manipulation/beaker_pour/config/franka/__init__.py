# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##



##
# Inverse Kinematics - Absolute Pose Control
##

# gym.register(
#     id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
#     },
#     disable_env_checker=True,
# )

##
# Inverse Kinematics - Relative Pose Control
##


# gym.register(
#     id="Dev-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.dev_env_cfg:FrankaDevEnvCfg",
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Pour-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pour_ik_rel_env_cfg:FrankaPourEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
        "robomimic_bc_trans_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_trans.json"),
        "robomimic_hbc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/hbc.json"),
        "robomimic_bcq_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bcq.json"),
        #"robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
# gym.register(
#     id="Dev-IK-Rel-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.dev_ik_rel_env_obs:FrankaDevEnvCfg",
#         "robomimic_bc_rnn_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
#         "robomimic_bc_trans_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_trans.json"),
#         "robomimic_hbc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/hbc.json"),
#         "robomimic_bcq_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bcq.json"),
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
#     },
#     disable_env_checker=True,
# )

