import argparse
import copy
import gymnasium as gym
import numpy as np
import random
import torch
import h5py  # <--- Added for writing dataset files

from isaaclab.app import AppLauncher

# Keep your original argparse block here...

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=500, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

# ... [Keep all your original parser arguments exactly as they are] ...
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from isaaclab_tasks.utils import parse_env_cfg


def rollout_and_collect(policy, env, success_term, horizon, device):
    """Perform a rollout and capture complete data matrices for failure detection."""
    policy.start_episode()
    obs_dict, _ = env.reset()
    
    # Initialize containers for sequential timestep data
    traj_data = {
        "actions": [],
        "rnn_hidden_states": [], # Track internal LSTM features
        "rnn_cell_states": [],
        "obs": {k: [] for k in policy.policy.obs_shapes.keys()}
    }

    success = False

    for i in range(horizon):
        # Clean & squeeze observations matching policy tracking expectations
        current_obs = {k: torch.squeeze(v) for k, v in obs_dict["policy"].items()
                       if k in policy.policy.obs_shapes}
        
        # Save observations BEFORE taking the step
        for k in traj_data["obs"].keys():
            # Convert tensor to numpy for structural storage
            traj_data["obs"][k].append(current_obs[k].detach().cpu().numpy())

        # Extract internal RNN layers state before forward pass updates them
        # Robomimic policies generally keep state in policy.policy.nets["policy"].rnn_state
        # Or implicitly within the policy framework's step memory wrapper.
        try:
            # Accessing the underlying LSTM state list [hidden_state, cell_state]
            rnn_state = policy.policy.get_rnn_state() 
            # Shapes: [num_layers, batch, hidden_dim] -> e.g., (2, 1, 400)
            traj_data["rnn_hidden_states"].append(rnn_state[0].detach().cpu().numpy().squeeze(1))
            traj_data["rnn_cell_states"].append(rnn_state[1].detach().cpu().numpy().squeeze(1))
        except AttributeError:
            # Fallback placeholder if state is managed implicitly or obscured
            pass

        # Compute actions
        actions = policy(current_obs)

        # Unnormalize actions (your original mapping code)
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = ((actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)) / 2 + args_cli.norm_factor_min
        
        # Save executed action
        traj_data["actions"].append(actions.flatten())

        actions_tensor = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Step Environment
        obs_dict, _, terminated, truncated, _ = env.step(actions_tensor)

        # Evaluate task termination statuses
        if bool(success_term.func(env, **success_term.params)[0]):
            success = True
            break
        elif terminated or truncated:
            success = False
            break

    # Convert lists into cohesive numpy arrays for serialization
    final_traj = {
        "actions": np.array(traj_data["actions"], dtype=np.float32),
        "obs": {k: np.array(v, dtype=np.float32) for k, v in traj_data["obs"].items()},
    }
    if len(traj_data["rnn_hidden_states"]) > 0:
        final_traj["rnn_hidden_states"] = np.array(traj_data["rnn_hidden_states"], dtype=np.float32)
        final_traj["rnn_cell_states"] = np.array(traj_data["rnn_cell_states"], dtype=np.float32)

    return success, final_traj


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.eval_mode = True
    env_cfg.eval_type = "vanilla"
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # File handle to save output trajectories for training FAIL-detect
    output_hdf5_path = f"docs/fail_detect_data_{args_cli.task}.hdf5"
    f_out = h5py.File(output_hdf5_path, "w")
    data_grp = f_out.create_group("data")

    results = []
    
    # Run rollouts to gather balanced telemetry data
    print(f"[INFO] Collecting datasets over {args_cli.num_rollouts} runs...")
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
        
        success, traj = rollout_and_collect(policy, env, success_term, args_cli.horizon, device)
        results.append(success)
        
        # Save directly to HDF5 structure matching typical mimicking styles
        demo_grp = data_grp.create_group(f"demo_{trial}")
        demo_grp.create_dataset("actions", data=traj["actions"])
        demo_grp.attrs["success"] = int(success) # 1 if successful, 0 if failure
        
        # Save features/hidden states if extracted
        if "rnn_hidden_states" in traj:
            demo_grp.create_dataset("rnn_hidden_states", data=traj["rnn_hidden_states"])
            demo_grp.create_dataset("rnn_cell_states", data=traj["rnn_cell_states"])

        # Save nested low-dim observation fields
        obs_grp = demo_grp.create_group("obs")
        for k, v in traj["obs"].items():
            obs_grp.create_dataset(k, data=v)

        print(f"[INFO] Trial {trial} Finished. Success Status: {success}\n")

    f_out.close()
    print(f"[INFO] Collection Complete! Data stored in: {output_hdf5_path}")
    print(f"Success metrics: {results.count(True)} successful runs out of {len(results)} total.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()