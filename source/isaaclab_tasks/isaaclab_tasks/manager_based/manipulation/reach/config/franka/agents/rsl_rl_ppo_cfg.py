# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
import sys
from pathlib import Path
tarloco_path = str(Path(__file__).resolve().parents[6] / "docs" / "TARLoco" / "exts" / "tarloco")
if tarloco_path not in sys.path:
    sys.path.append(tarloco_path)
#from exts.tarloco.tasks.agents.rsl_rl_cfg import Go1RoughRnnTarRunnerCfg
# Import TAR policy and algorithm configs from TARLoco
try:
    from tarloco.tasks.agents.rsl_rl_cfg import (
        RslRlPpoPolicyCfg,
        RslRlRnnPpoPolicyCfg,
        RslRlRnnTarPolicyCfg,
        ppo_algo_cfg,
        tar_algo_cfg,
    )
except ImportError:
    # Fallback definition for standard algorithm config if tarloco imports differ
    ppo_algo_cfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    tar_algo_cfg = ppo_algo_cfg


@configclass
class FrankaReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "franka_reach"
    run_name = ""
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = ppo_algo_cfg


@configclass
class Go1RoughPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "TAR_workspace"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = ppo_algo_cfg


@configclass
class Go1RoughRnnRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "TAR_workspace"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = ppo_algo_cfg


@configclass
class Go1RoughRnnTarRunnerCfg(Go1RoughRnnRunnerCfg):
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = tar_algo_cfg