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


#from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cube_lift.config.franka import dev_ik_rel_env_place_vismot
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
parser.add_argument(
    "--log_file", type=str, default="docs/rollout_analysis.txt", help="Path to text file for recording initial positions and outcomes."
)

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


import torch

def print_obs_diagnostics(obs_dict, label="RAW ROLLOUT OBS"):
    """Prints tensor metadata to verify shapes, channel counts, and value ranges."""
    print(f"\n================ [{label}] DIAGNOSTICS ================")
    # Unnest policy dict if present
    data = obs_dict.get("policy", obs_dict) if isinstance(obs_dict, dict) else obs_dict

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            shape_str = str(tuple(val.shape))
            min_val = val.min().item() if val.numel() > 0 else float('nan')
            max_val = val.max().item() if val.numel() > 0 else float('nan')
            print(f"Key: {key:20s} | Shape: {shape_str:18s} | Dtype: {val.dtype} | Min: {min_val:7.3f} | Max: {max_val:7.3f}")

            # Check image dimension assumptions
            if val.ndim in (3, 4):
                c_last = val.shape[-1]
                c_first = val.shape[1] if val.ndim == 4 else val.shape[0]
                
                if c_last == 4 or c_first == 4:
                    print(f"   ⚠️ RGBA DETECTED (4 channels). Checks for 3 channels (v.shape[-1] == 3) will evaluate to FALSE!")
                elif c_last == 3 or c_first == 3:
                    print(f"   ✓ 3-Channel RGB detected.")
    print("========================================================\n")


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


# def process_obs(obs_dict, expected_shapes):
#     """Process observations to match expected shapes by the policy."""
#     processed = {}
#     for k, v in obs_dict.items():
#         if k not in expected_shapes:
#             continue

#         expected_shape = expected_shapes[k]

#         # If it has a batch dimension [1, ...], squeeze it
#         if v.ndim > len(expected_shape):
#             v = v.squeeze(0)

#         # Handle images (3D tensors)
#         if v.ndim == 3 and (v.shape[-1] == 3 or v.shape[0] == 3):
#             # Ensure input is CHW format for PyTorch / Robomimic
#             if v.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
#                 v = v.permute(2, 0, 1)

#             target_res = (expected_shape[1], expected_shape[2])

#             # Resize if H, W don't match expected resolution
#             if v.shape[1:] != target_res:
#                 v_float = v.unsqueeze(0).float()  # (1, C, H, W)
#                 v_resized = F.interpolate(
#                     v_float, size=target_res, mode="bilinear", align_corners=False
#                 )
#                 v = v_resized.squeeze(0)  # (C, H, W)
#                 if obs_dict[k].dtype == torch.uint8:
#                     v = v.to(torch.uint8)
#         else:
#             v=v.float()
#             # # Low-dim normalization using dataset stats
#             # norm_stats = {
#             #     "eef_pos": {"mean": [0.181, -0.1197, 0.2355], "std": [0.0768, 0.171, 0.0994]},
#             #     "eef_quat": {"mean": [0.4927, 0.5042, -0.4988, -0.4976], "std": [0.0345, 0.0333, 0.0458, 0.0463]},
#             #     "gripper_pos": {"mean": [-0.2128, 0.2139], "std": [0.1793, 0.1804]},
#             #     "object_position": {"mean": [0.1687, -0.1263, 0.1264], "std": [0.0416, 0.1663, 0.0766]},
#             #     "target_object_position": {"mean": [0.9362, -1.0449, 0.0126], "std": [2.5137, 1.9732, 0.0007]}
#             # }

#             # if k in norm_stats:
#             #     mean = torch.tensor(norm_stats[k]["mean"], dtype=torch.float32, device=v.device)
#             #     std = torch.tensor(norm_stats[k]["std"], dtype=torch.float32, device=v.device)
#             #     std = torch.clamp(std, min=1e-6)
#             #     v = (v - mean) / std

#         processed[k] = v
#     return processed

# def process_obs(obs_dict, expected_shapes):
#     """Process observations to match expected shapes by the policy."""
#     processed = {}
#     for k, v in obs_dict.items():
#         if k not in expected_shapes:
#             continue

#         expected_shape = expected_shapes[k]

#         # Squeeze batch dimension [1, ...] if present
#         if v.ndim > len(expected_shape):
#             v = v.squeeze(0)

#         # Handle images (3D tensors)
#         if v.ndim == 3 and (v.shape[-1] == 3 or v.shape[0] == 3):
#             # Ensure input is CHW format
#             if v.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
#                 v = v.permute(2, 0, 1)

#             target_res = (expected_shape[1], expected_shape[2])

#             # Resize if needed
#             if v.shape[1:] != target_res:
#                 v_float = v.unsqueeze(0).float()
#                 v = F.interpolate(
#                     v_float, size=target_res, mode="bilinear", align_corners=False
#                 ).squeeze(0)

#             # Ensure image tensors are float32 in [0, 1] range
#             v = v.float()
#             if v.max() > 1.0:
#                 v = v / 255.0
#         else:
#             # Low-dim observations
#             v = v.float()

#         processed[k] = v
#     return processed

def process_obs(obs_dict, expected_shapes):
    """Process observations for direct policy.policy inference."""
    processed = {}
    for k, v in obs_dict.items():
        #print_obs_diagnostics({k: v}, label=f"INSIDE PROCESS_OBS ({k})")
        if k not in expected_shapes:
            continue

        expected_shape = expected_shapes[k]

        # Squeeze leading batch dimension if present [1, ...] -> [...]
        if v.ndim > len(expected_shape):
            v = v.squeeze(0)

        # Handle images (3D tensors)
        if v.ndim == 3 and (v.shape[-1] == 3 or v.shape[0] == 3):
            # Ensure CHW format (C, H, W) for PyTorch neural nets
            if v.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
                v = v.permute(2, 0, 1)

            target_res = (expected_shape[1], expected_shape[2])

            # Resize if H, W don't match expected resolution
            if v.shape[1:] != target_res:
                v_float = v.unsqueeze(0).float()
                v = F.interpolate(
                    v_float, size=target_res, mode="bilinear", align_corners=False
                ).squeeze(0)

            # Convert to float and scale to [0.0, 1.0]
            v = v.float()
            if v.max() > 1.0:
                v = v / 255.0
        else:
            # Low-dim vector observations (ensure float32, keep raw values)
            v = v.float()
        
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
    # if not hasattr(prepare_obs_for_diffusion, '_printed'):
    #     print("\n[DEBUG] Observation shapes after stacking:")
    #     for k, v in stacked_obs.items():
    #         print(f"  {k}: {v.shape} (ndim={v.ndim})")
    #     #prepare_obs_for_diffusion._printed = True
    
    return stacked_obs


# # def rollout(policy, env, success_term, horizon, device):
#     """Perform a single rollout of the diffusion policy in the environment.

#     Args:
#         policy: The robomimic diffusion policy to play.
#         env: The environment to play in.
#         horizon: The step horizon of each rollout.
#         device: The device to run the policy on.

#     Returns:
#         terminated: Whether the rollout terminated successfully.
#         traj: The trajectory of the rollout.
#     """
#     policy.start_episode()
#     obs_dict, _ = env.reset()
#     traj = dict(actions=[], obs=[], next_obs=[])

#     # Setup temporal observation history for diffusion policy
#     observation_horizon = get_observation_horizon(policy)
#     print(f"[INFO] Using observation_horizon={observation_horizon} for rollout")
#     obs_history = deque(maxlen=observation_horizon)

#     # Diagnose image dtype and range on first step
#     if "table_cam" in obs_dict["policy"]:
#         img = obs_dict["policy"]["table_cam"]
#         print(f"[DEBUG] table_cam: dtype={img.dtype}, shape={img.shape}, "
#               f"min={img.min().item():.3f}, max={img.max().item():.3f}")
    
#     # Initialize history with first observation (repeated to fill the queue)
#     # Process and filter observations to match policy expectations
    
#     first_obs = process_obs(obs_dict["policy"], policy.policy.obs_shapes)
    
#     # Fill history with copies of first observation
#     for _ in range(observation_horizon):
#         obs_history.append(copy.deepcopy(first_obs))
#     action_queue = deque()
#     action_horizon = policy.policy.algo_config.horizon.action_horizon
#     for i in range(horizon):
#         # 1. Replenish action queue when empty
#         if len(action_queue) == 0:
#             obs = prepare_obs_for_diffusion(obs_dict["policy"], obs_history, policy.policy.obs_shapes)
#             traj["obs"].append(obs)

#             action_sequence = policy.policy._get_action_trajectory(obs_dict=obs)

#             if isinstance(action_sequence, np.ndarray):
#                 action_sequence = torch.from_numpy(action_sequence).to(device)

#             if action_sequence.ndim == 2:
#                 action_sequence = action_sequence.unsqueeze(0)

#             num_actions_to_queue = min(action_horizon, action_sequence.shape[1])
#             for t in range(num_actions_to_queue):
#                 action_queue.append(action_sequence[0, t])

#             print(f"\n[QUEUE REPLENISHED @ Step {i}]")
#             print(f"  -> Generated sequence shape: {list(action_sequence.shape)}")
#             print(f"  -> Queued {len(action_queue)} action vectors (Action Horizon = {action_horizon})")

#         # 2. Pop next action vector
#         actions = action_queue.popleft()

#         if isinstance(actions, np.ndarray):
#             actions = torch.from_numpy(actions).to(device)

#         # 3. Optional action un-normalization
#         if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
#             actions = (
#                 (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
#             ) / 2 + args_cli.norm_factor_min

#         # 4. Reshape for environment step [1, action_dim]
#         actions = actions.view(1, env.action_space.shape[1])

#         # 5. Single environment step
#         obs_dict, _, terminated, truncated, _ = env.step(actions)

#         # Record trajectory
#         traj["actions"].append(actions.tolist())
#         traj["next_obs"].append(obs_dict["policy"])

#         # Check success/termination
#         if bool(success_term.func(env, **success_term.params)[0]):
#             return True, traj
#         elif terminated or truncated:
#             return False, traj

#     return False, traj
def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the diffusion policy in the environment."""
    policy.start_episode()
    obs_dict, _ = env.reset()
    #print_obs_diagnostics(obs_dict, label="INITIAL ENV RESET")
    traj = dict(actions=[], obs=[], next_obs=[])
    # log start position
    initial_policy_obs = obs_dict["policy"]
    init_obj_pos = initial_policy_obs["object_position"].detach().cpu().numpy().flatten().tolist()
    init_target_pos = initial_policy_obs["target_object_position"].detach().cpu().numpy().flatten().tolist()
    # Setup temporal observation history for diffusion policy
    observation_horizon = get_observation_horizon(policy)
    print(f"[INFO] Using observation_horizon={observation_horizon} for rollout")
    obs_history = deque(maxlen=observation_horizon)

    # Process and fill history with copies of initial observation
    first_obs = process_obs(obs_dict["policy"], policy.policy.obs_shapes)
    for _ in range(observation_horizon):
        obs_history.append(copy.deepcopy(first_obs))

    action_queue = deque()
    action_horizon = 12#  policy.policy.algo_config.horizon.action_horizon
    print(f"[INFO] Using action horizon ={action_horizon} for rollout")
    for i in range(horizon):
    
        # Replenish action queue when empty
        if len(action_queue) == 0:
            obs = {}
            for k in first_obs:
                obs_list = [obs_history[j][k] for j in range(len(obs_history))]
                obs[k] = torch.stack(obs_list, dim=0).unsqueeze(0)
           # print(f"policy_obs: {obs}")
            # 1. Get normalized action predictions from diffusion policy
            action_sequence = policy.policy._get_action_trajectory(obs_dict=obs)
           # print(action_sequence.shape)
            if isinstance(action_sequence, np.ndarray):
                action_sequence = torch.from_numpy(action_sequence).to(device)

            if action_sequence.ndim == 2:
                action_sequence = action_sequence.unsqueeze(0)

            # 2. Un-normalize actions using Robomimic's internal dataset stats
            if hasattr(policy.policy, "action_normalization_stats") and policy.policy.action_normalization_stats is not None:
                print(f"[INFO] Unnormalising...")
                stats = policy.policy.action_normalization_stats
                # Robomimic standard min/max or mean/std unnormalization
                if "min" in stats and "max" in stats:
                    a_min = torch.tensor(stats["min"], device=device)
                    a_max = torch.tensor(stats["max"], device=device)
                    # Map [-1, 1] -> [min, max]
                    action_sequence = 0.5 * (action_sequence + 1.0) * (a_max - a_min) + a_min
                elif "mean" in stats and "std" in stats:
                    a_mean = torch.tensor(stats["mean"], device=device)
                    a_std = torch.tensor(stats["std"], device=device)
                    action_sequence = action_sequence * a_std + a_mean

            # 3. Queue un-normalized actions
            num_actions_to_queue = min(action_horizon, action_sequence.shape[1])
            for t in range(num_actions_to_queue):
                action_queue.append(action_sequence[0, t])
           # print(f"\n[QUEUE REPLENISHED @ Step {i}]")
           # print(f"  -> Generated sequence shape: {list(action_sequence.shape)}")
           # print(f"  -> Queued {len(action_queue)} action vectors (Action Horizon = {action_horizon})")

        # Pop next action vector
        actions = action_queue.popleft()
        #print(f"step {i}, action: {actions}")


        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(device)

        #  action un-normalization
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        # Reshape for environment step [1, action_dim]
        actions = actions.view(1, env.action_space.shape[1])

        

        
        obs_dict, _, terminated, truncated, info = env.step(actions)
        #print(f"Step: {env.step(actions)}")
        # update observation history on EVERY step
        curr_processed_obs = process_obs(obs_dict["policy"], policy.policy.obs_shapes)
        obs_history.append(curr_processed_obs)

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs_dict["policy"])

        is_success = bool(success_term.func(env, **success_term.params)[0])
        is_terminated = bool(terminated.item() if hasattr(terminated, "item") else terminated)
        is_truncated = bool(truncated.item() if hasattr(truncated, "item") else truncated)
        did_pickup = torch.any(obs_dict["subtask"]["grasp"])
        did_appr_goal = torch.any(obs_dict["subtask"]["appr_goal"])
        if is_success:
            return True, traj, "finished", init_obj_pos, init_target_pos, True, True
        
        elif is_terminated:
            log_data = info.get("log", {})
            reasons = []
            for key, val in log_data.items():
                if key.startswith("Episode_Termination/"):
                    val_num = val.item() if hasattr(val, "item") else val
                    if val_num > 0:
                        term_name = key.replace("Episode_Termination/", "")
                        reasons.append(term_name)
            
            reason = ", ".join(reasons) if reasons else "Task Failure (Terminated)"
            return False, traj, reason, init_obj_pos, init_target_pos, did_pickup.item(), did_appr_goal.item()

        elif is_truncated or (i == horizon - 1):
            return False, traj, "timeout", init_obj_pos, init_target_pos, did_pickup.item(), did_appr_goal.item()

    return False, traj, "idk", init_obj_pos, init_target_pos, False, False

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
    log_file = args_cli.log_file
    with open(log_file, "w") as f:
        f.write("Trial\tSuccess\tReason\tObj_X\tObj_Y\tObj_Z\tTarget_X\tTarget_Y\tTarget_Z\n")
    print(f"[INFO] Initialized position logging to: {log_file}")
    # Run policy rollouts
    results = []
    reasons = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        terminated, traj, reason, init_obj_pos, init_target_pos, grasp, appr_goal = rollout(
            policy, env, success_term, args_cli.horizon, device
        )
        results.append(terminated)
        reasons.append(reason)

        # Format initial positions as tab-separated values
        obj_str = "\t".join([f"{x:.4f}" for x in init_obj_pos])
        target_str = "\t".join([f"{x:.4f}" for x in init_target_pos])
        
        # Append rollout analysis data to text file
        log_line = f"{trial}\t{terminated}\t{reason}\t{obj_str}\t{target_str}\n"
        with open(log_file, "a") as f:
            f.write(log_line)

        print(f"[INFO] Trial {trial}: {'Success' if terminated else 'Failed:'} {reason}")
        print(f"       Init Object Pos: {[round(x, 4) for x in init_obj_pos]}")
        print(f"       Init Target Pos: {[round(x, 4) for x in init_target_pos]}\n")
        print(f"       Grasp: {grasp} , Appr Goal: {appr_goal}\n")
        print(f"[INFO] Trial {trial}: {'Success' if terminated else 'Failed :' } {reason}\n")
        print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results):.2%}")
    print(f"Trial Results: {results}\n")
    print(f"Termination Conditions: {reasons}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
