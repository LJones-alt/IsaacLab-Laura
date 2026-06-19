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
parser.add_argument("--horizon", type=int, default=400, help="Step horizon of each rollout.")
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
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def get_observation_horizon(policy):
    """Get the observation horizon from diffusion policy config."""
    # Try algo_config first (robomimic's internal config object)
    if hasattr(policy.policy, 'algo_config') and hasattr(policy.policy.algo_config, 'horizon'):
        h = policy.policy.algo_config.horizon.observation_horizon
        print(f"[DEBUG] observation_horizon from algo_config: {h}")
        return h
    # Try global_config
    if hasattr(policy.policy, 'global_config'):
        try:
            h = policy.policy.global_config.algo.horizon.observation_horizon
            print(f"[DEBUG] observation_horizon from global_config: {h}")
            return h
        except AttributeError:
            pass
    print("[WARNING] Could not find observation_horizon, defaulting to 2")
    return 2


def process_obs(obs_dict, expected_shapes):
    """Process observations to match expected shapes by the policy.
    
    Handles:
    - Resizing images to match expected resolution while keeping HWC format
    """
    processed = {}
    for k, v in obs_dict.items():
        if k not in expected_shapes:
            continue
            
        expected_shape = expected_shapes[k]
        
        # If it has a batch dimension [1, ...], squeeze it
        if v.ndim > len(expected_shape):
            v = v.squeeze(0)
            
        # Handle images (3D tensors with channels in last dim)
        if v.ndim == 3 and v.shape[-1] == 3:
            # Isaac Lab: (H, W, C). Target resolution is (expected_shape[1], expected_shape[2])
            target_res = (expected_shape[1], expected_shape[2])
            
            # Resize if dimensions don't match
            if v.shape[:2] != target_res:
                # Interpolate expects 4D input: (B, C, H, W)
                v_float = v.permute(2, 0, 1).unsqueeze(0).float()
                v_resized = F.interpolate(
                    v_float, size=target_res, mode="bilinear", align_corners=False
                )
                # Permute back to (H, W, C) for robomimic internal processing
                v = v_resized.squeeze(0).permute(1, 2, 0)
                # Convert back to original dtype if it was uint8
                if obs_dict[k].dtype == torch.uint8:
                    v = v.to(torch.uint8)
        
        processed[k] = v
    return processed


def prepare_obs_for_diffusion(obs_dict, obs_history, expected_shapes):
    """Prepare observations for diffusion policy inference.
    
    Diffusion policy expects observations with shape [B, T, D] where:
    - B = batch size (1 for single env inference)
    - T = observation_horizon (temporal dimension)
    - D = observation feature dimension
    
    Args:
        obs_dict: Current observation dictionary from environment
        obs_history: Deque of past observations
        expected_shapes: Dictionary of expected shapes from policy
        
    Returns:
        Stacked observation dictionary with temporal dimension
    """
    # Process current observation to match expected format
    current_obs = process_obs(obs_dict, expected_shapes)
    
    obs_history.append(current_obs)
    
    # Stack observations along temporal dimension: [T, D] -> [1, T, D]
    stacked_obs = {}
    for k in current_obs:
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
    print(f"[INFO] Using observation_horizon={observation_horizon} for rollout")
    obs_history = deque(maxlen=observation_horizon)

    # Diagnose image dtype and range on first step
    if "table_cam" in obs_dict["policy"]:
        img = obs_dict["policy"]["table_cam"]
        print(f"[DEBUG] table_cam: dtype={img.dtype}, shape={img.shape}, "
              f"min={img.min().item():.3f}, max={img.max().item():.3f}")
    
    # Initialize history with first observation (repeated to fill the queue)
    # Process and filter observations to match policy expectations
    
    first_obs = process_obs(obs_dict["policy"], policy.policy.obs_shapes)
    
    # Fill history with copies of first observation
    for _ in range(observation_horizon):
        obs_history.append(copy.deepcopy(first_obs))

    for i in range(horizon):
        # Stack temporally for diffusion policy: history gives [1, T, D] tensors
        obs = prepare_obs_for_diffusion(obs_dict["policy"], obs_history, policy.policy.obs_shapes)

        traj["obs"].append(obs)

        # Compute actions from diffusion policy
        # Use batched_ob=True because we already have batch dimension [1, T, D]
        actions = policy(obs, batched_ob=True)

        # Diagnose action output on first few steps
        if i < 3:
            act_flat = actions.flatten().tolist()
            print(f"[DEBUG] Step {i} arm actions: {[f'{v:.3f}' for v in act_flat[:6]]}, gripper: {act_flat[6]:.4f}")

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
