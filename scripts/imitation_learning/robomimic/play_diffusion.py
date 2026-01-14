# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained Diffusion Policy from robomimic.

This script loads a robomimic diffusion policy and plays it in an Isaac Lab environment.
It handles the temporal observation stacking required by diffusion policy.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, override the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic diffusion policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
from collections import deque
import gymnasium as gym
import numpy as np
import random
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def get_observation_horizon(policy):
    """Get the observation horizon from diffusion policy config.
    
    Args:
        policy: The loaded robomimic policy wrapper.
        
    Returns:
        int: The observation horizon (default 2 for diffusion policy).
    """
    if hasattr(policy.policy, 'algo_config') and hasattr(policy.policy.algo_config, 'horizon'):
        return policy.policy.algo_config.horizon.observation_horizon
    # Default for diffusion policy if not found
    return 2


def prepare_obs_for_diffusion(obs_dict, obs_history):
    """Prepare observations for diffusion policy inference.
    
    Diffusion policy expects observations with shape [B, T, D] where:
    - B = batch size (1 for single env inference)
    - T = observation_horizon (temporal dimension)
    - D = observation feature dimension
    
    Args:
        obs_dict: Current observation dictionary from environment
        obs_history: Deque of past observations
        
    Returns:
        Stacked observation dictionary with temporal dimension
    """
    # Store current observation (without batch dimension)
    current_obs = {}
    for k in obs_dict:
        if obs_dict[k].dim() > 1:
            current_obs[k] = obs_dict[k].squeeze(0)
        else:
            current_obs[k] = obs_dict[k]
    
    obs_history.append(current_obs)
    
    # Stack observations along temporal dimension: [T, D] -> [1, T, D]
    stacked_obs = {}
    for k in obs_dict:
        obs_list = [obs_history[i][k] for i in range(len(obs_history))]
        stacked = torch.stack(obs_list, dim=0).unsqueeze(0)  # [1, T, D]
        stacked_obs[k] = stacked
    
    # Debug: print shapes on first call
    if not hasattr(prepare_obs_for_diffusion, '_printed'):
        print("\n[DEBUG] Observation shapes after stacking:")
        for k, v in stacked_obs.items():
            print(f"  {k}: {v.shape} (ndim={v.ndim})")
        prepare_obs_for_diffusion._printed = True
    
    return stacked_obs


def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the diffusion policy in the environment.

    Args:
        policy: The robomimic diffusion policy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated successfully.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    # Setup temporal observation history for diffusion policy
    observation_horizon = get_observation_horizon(policy)
    obs_history = deque(maxlen=observation_horizon)
    
    # Initialize history with first observation (repeated to fill the queue)
    first_obs = {}
    for k in obs_dict["policy"]:
        if obs_dict["policy"][k].dim() > 1:
            first_obs[k] = obs_dict["policy"][k].squeeze(0)
        else:
            first_obs[k] = obs_dict["policy"][k]
    
    # Fill history with copies of first observation
    for _ in range(observation_horizon):
        obs_history.append(copy.deepcopy(first_obs))

    for i in range(horizon):
        # Prepare observations with temporal stacking for diffusion policy
        obs = prepare_obs_for_diffusion(obs_dict["policy"], obs_history)

        # Debug: print gripper observation at key steps
        # if i == 0 or i == 5:
        #     print(f"\n[DEBUG] Step {i} - Observations:")
        #     print(f"  gripper_pos: {obs['gripper_pos'][0, -1, :].cpu().numpy()}")  # Last timestep
        #     print(f"  eef_pos: {obs['eef_pos'][0, -1, :].cpu().numpy()}")
        #     print(f"  object_position: {obs['object_position'][0, -1, :].cpu().numpy()}")

        # Check if environment has image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    # Stack image observations temporally as well
                    obs[image_name] = image.unsqueeze(0).unsqueeze(0).expand(1, observation_horizon, -1, -1, -1)

        traj["obs"].append(obs)

        # Compute actions from diffusion policy
        # Use batched_ob=True because we already have batch dimension [1, T, D]
        actions = policy(obs, batched_ob=True)

        # Debug: print action stats periodically
        # if i == 0 or i == 10 or i == 50:
        #     print(f"\n[DEBUG] Step {i} - Raw policy actions:")
        #     print(f"  Shape: {actions.shape}")
        #     print(f"  Min: {actions.min():.4f}, Max: {actions.max():.4f}")
        #     print(f"  Values: {actions}")
        #     # Check if policy has normalization stats
        #     if hasattr(policy, 'action_normalization_stats') and policy.action_normalization_stats is not None:
        #         print(f"  Action norm stats available: {policy.action_normalization_stats.keys()}")
        #     else:
        #         print(f"  WARNING: No action normalization stats in policy!")

        # Unnormalize actions if normalization factors provided
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions to environment
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs_dict["policy"])

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def main():
    """Run a trained diffusion policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed for reproducibility
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load policy once (diffusion policy maintains internal action queue)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
    print(f"[INFO] Loaded diffusion policy from {args_cli.checkpoint}")
    
    # Debug: print expected observation shapes from policy
    print("\n[DEBUG] Policy expected obs_shapes:")
    for k, v in policy.policy.obs_shapes.items():
        print(f"  {k}: {v} (len={len(v)})")
    
    # Get and print observation horizon
    obs_horizon = get_observation_horizon(policy)
    print(f"[INFO] Diffusion policy observation_horizon: {obs_horizon}")

    # Run policy rollouts
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        terminated, traj = rollout(policy, env, success_term, args_cli.horizon, device)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {'Success' if terminated else 'Failed'}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results):.2%}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
