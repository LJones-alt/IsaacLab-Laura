# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained policy from robomimic across multiple evaluation settings.

This script loads a trained robomimic policy and evaluates it in an Isaac Lab environment
across multiple evaluation settings (lighting, textures, etc.) and seeds. It saves the results
to a specified output directory.

Args:
    task: Name of the environment.
    input_dir: Directory containing the model checkpoints to evaluate.
    horizon: Step horizon of each rollout.
    num_rollouts: Number of rollouts per model per setting.
    num_seeds: Number of random seeds to evaluate.
    seeds: Optional list of specific seeds to use instead of random ones.
    log_dir: Directory to write results to.
    log_file: Name of the output file.
    output_vis_file: File path to export recorded episodes.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
    disable_fabric: Whether to disable fabric and use USD I/O operations.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--input_dir", type=str, default=None, help="Directory containing models to evaluate.")
parser.add_argument(
    "--start_epoch", type=int, default=100, help="Epoch of the checkpoint to start the evaluation from."
)
parser.add_argument("--horizon", type=int, default=400, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=15, help="Number of rollouts for each setting.")
parser.add_argument("--num_seeds", type=int, default=3, help="Number of random seeds to evaluate.")
parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of specific seeds to use.")
parser.add_argument(
    "--log_dir", type=str, default="/tmp/policy_evaluation_results", help="Directory to write results to."
)
parser.add_argument("--log_file", type=str, default="results", help="Name of output file.")
parser.add_argument(
    "--output_vis_file", type=str, default="visuals.hdf5", help="File path to export recorded episodes."
)
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
    # Import pinocchio before AppLauncher to force the use of the version installed
    # by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import os
import pathlib
import random
import re
from collections import deque
import numpy as np

import gymnasium as gym
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

from isaaclab_tasks.utils import parse_env_cfg


def get_observation_horizon(policy):
    """Get the observation horizon from the policy config."""
    # Diffusion policy specific
    if hasattr(policy.policy, "algo_config") and hasattr(policy.policy.algo_config, "horizon"):
        return policy.policy.algo_config.horizon.observation_horizon
    # Fallback for standard policies
    return 1


def prepare_obs_for_diffusion(obs_dict, obs_history):
    """Prepare observations for diffusion policy inference.
    
    Diffusion policy expects observations with shape [B, T, D] where:
    - B = batch size (1 for single env inference)
    - T = observation_horizon (temporal dimension)
    - D = observation feature dimension
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
    
    return stacked_obs


def rollout(policy, env: gym.Env, success_term, horizon: int, device: torch.device) -> tuple[bool, dict]:
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimic policy to evaluate.
        env: The environment to evaluate in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.
        args_cli: Command line arguments containing normalization factors.

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
    # Filter observations to only include keys the policy was trained on
    filtered_first_obs_dict = {k: v for k, v in obs_dict["policy"].items() if k in policy.policy.obs_shapes}
    
    first_obs = {}
    for k in filtered_first_obs_dict:
        if filtered_first_obs_dict[k].dim() > 1:
            first_obs[k] = filtered_first_obs_dict[k].squeeze(0)
        else:
            first_obs[k] = filtered_first_obs_dict[k]
    
    # Fill history with copies of first observation
    for _ in range(observation_horizon):
        obs_history.append(copy.deepcopy(first_obs))

    for _ in range(horizon):
        # Filter observations to only include keys the policy was trained on
        # robomimic handles image processing internally (HWC uint8 -> normalization)
        filtered_obs_dict = {k: v for k, v in obs_dict["policy"].items() if k in policy.policy.obs_shapes}

        if observation_horizon > 1:
            # Stack temporally for diffusion policy: history gives [1, T, D] tensors
            obs = prepare_obs_for_diffusion(filtered_obs_dict, obs_history)
            batched_ob = True
        else:
            # Standard policies expect [D] and we let robomimic add batch dim
            obs = copy.deepcopy(filtered_obs_dict)
            for ob in obs:
                obs[ob] = torch.squeeze(obs[ob])
            batched_ob = False

        traj["obs"].append(obs)

        # Compute actions
        actions = policy(obs, batched_ob=batched_ob)
       # print(f"Actions: {actions}")
        # Unnormalize actions if normalization factors are provided
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def evaluate_model(
    model_path: str,
    env: gym.Env,
    device: torch.device,
    success_term,
    num_rollouts: int,
    horizon: int,
    seed: int,
    output_file: str,
) -> float:
    """Evaluate a single model checkpoint across multiple rollouts.

    Args:
        model_path: Path to the model checkpoint.
        env: The environment to evaluate in.
        device: The device to run the policy on.
        num_rollouts: Number of rollouts to perform.
        horizon: Step horizon of each rollout.
        seed: Random seed to use.
        output_file: File to write results to.

    Returns:
        float: Success rate of the model
    """
    # Set seed
    torch.manual_seed(seed)
    env.seed(seed)
    random.seed(seed)

    # Load policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=model_path, device=device, verbose=False)
    print(f"[DEBUG] Policy : {policy}")
    # Run policy
    results = []
    for trial in range(num_rollouts):
        print(f"[Model: {os.path.basename(model_path)}] Starting trial {trial}")
        terminated, _ = rollout(policy, env, success_term, horizon, device)
        results.append(terminated)
        with open(output_file, "a") as file:
            file.write(f"[Model: {os.path.basename(model_path)}] Trial {trial}: {terminated}\n")
        print(f"[Model: {os.path.basename(model_path)}] Trial {trial}: {terminated}")

    # Calculate and log results
    success_rate = results.count(True) / len(results)
    with open(output_file, "a") as file:
        file.write(
            f"[Model: {os.path.basename(model_path)}] Successful trials: {results.count(True)}, out of"
            f" {len(results)} trials\n"
        )
        file.write(f"[Model: {os.path.basename(model_path)}] Success rate: {success_rate}\n")
        file.write(f"[Model: {os.path.basename(model_path)}] Results: {results}\n")
        file.write("-" * 80 + "\n\n")

    print(
        f"\n[Model: {os.path.basename(model_path)}] Successful trials: {results.count(True)}, out of"
        f" {len(results)} trials"
    )
    print(f"[Model: {os.path.basename(model_path)}] Success rate: {success_rate}\n")
    print(f"[Model: {os.path.basename(model_path)}] Results: {results}\n")

    return success_rate


def main() -> None:
    """Run evaluation of trained policies from robomimic with Isaac Lab environment."""
    # Parse configuration
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

    # Set evaluation settings
    env_cfg.eval_mode = True

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=False)

    # Get model checkpoints
    model_checkpoints = [f.name for f in os.scandir(args_cli.input_dir) if f.is_file()]

    # Set up seeds
    seeds = random.sample(range(0, 10000), args_cli.num_seeds) if args_cli.seeds is None else args_cli.seeds

    # Define evaluation settings
    settings = ["vanilla", "light_intensity", "light_color", "light_texture", "table_texture", "robot_texture", "all"]

    # Create log directory if it doesn't exist
    os.makedirs(args_cli.log_dir, exist_ok=True)

    # Evaluate each seed
    for seed in seeds:
        output_path = os.path.join(args_cli.log_dir, f"{args_cli.log_file}_seed_{seed}")
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize results summary
        results_summary = dict()
        results_summary["overall"] = {}
        for setting in settings:
            results_summary[setting] = {}

        with open(output_path, "w") as file:
            # Evaluate each setting
            for setting in settings:
                env.cfg.eval_type = setting

                file.write(f"Evaluation setting: {setting}\n")
                file.write("=" * 80 + "\n\n")

                print(f"Evaluation setting: {setting}")
                print("=" * 80)
                print(f"Now starting to evaluate model")
                # Evaluate each model
                for model in model_checkpoints:
                    # Skip files that are not checkpoints
                    if not model.endswith((".pth", ".ckpt")):
                        continue

                    # Skip early checkpoints
                    try:
                        # Try to find "epoch_X" in the filename
                        match = re.search(r"epoch_(\d+)", model)
                        if match:
                            model_epoch = int(match.group(1))
                        else:
                            # Fallback to the old logic if "epoch_" not found
                            model_epoch = int(model.split(".")[0].split("_")[-1])
                    except (ValueError, IndexError):
                        # If parsing fails, assume epoch 0
                        model_epoch = 0

                    if model_epoch < args_cli.start_epoch:
                        print(f"[INFO] Skipping model {model} (epoch {model_epoch} < {args_cli.start_epoch})")
                        continue

                    model_path = os.path.join(args_cli.input_dir, model)
                    success_rate = evaluate_model(
                        model_path=model_path,
                        env=env,
                        device=device,
                        success_term=success_term,
                        num_rollouts=args_cli.num_rollouts,
                        horizon=args_cli.horizon,
                        seed=seed,
                        output_file=output_path,
                    )

                    # Store results
                    results_summary[setting][model] = success_rate
                    if model not in results_summary["overall"].keys():
                        results_summary["overall"][model] = 0.0
                    results_summary["overall"][model] += success_rate

                    env.reset()

                file.write("=" * 80 + "\n\n")
                env.reset()

            # Calculate overall success rates
            for model in results_summary["overall"].keys():
                results_summary["overall"][model] /= len(settings)

            # Write final summary
            file.write("\nResults Summary (success rate):\n")
            for setting in results_summary.keys():
                file.write(f"\nSetting: {setting}\n")
                for model in results_summary[setting].keys():
                    file.write(f"{model}: {results_summary[setting][model]}\n")
                
                if results_summary[setting]:
                    max_key = max(results_summary[setting], key=results_summary[setting].get)
                    file.write(
                        f"\nBest model for setting {setting} is {max_key} with success rate"
                        f" {results_summary[setting][max_key]}\n"
                    )
                else:
                    file.write(f"\nNo models were evaluated for setting {setting}\n")

        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()