# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# run this script: 
# ./isaaclab.sh -p scripts/imitation_learning/robomimic/play_ensemble.py --device cuda --task Dev-IK-Rel-v0 --num_rollouts 100 
# stack cube task: ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --device cuda --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_rollouts 50
"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=800, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=2000, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

parser.add_argument("--model_name", type=str, default=None)

group = parser.add_mutually_exclusive_group()
group.add_argument("--use_recovery", type=bool ,default= False, help="Enable recovery mechanism. By default recovery is enabled.")

parser.add_argument("--ensemble_size", type=int, default=10)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
import numpy as np
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import copy
import random
import os
import gymnasium as gym
import torch
import math
from collections import deque

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from isaaclab_tasks.manager_based.manipulation.cube_lift.mdp.observations import get_joint_pos, object_grasped
#from chills.tasks.mdp.observations import get_joint_pos, object_grasped
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
#from isaaclab.utils.logging_helper import LoggingHelper, ErrorType, LogType
from isaaclab.utils.math import matrix_from_quat, convert_quat, quat_mul, euler_xyz_from_quat
from evaluation import ensemble_uncertainty
from isaaclab.safety.switchingLogic import SwitchingLogic
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from backup_controller_handler import BackupController 

def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    
    #print(f"obs_dict type: {type(obs_dict)},\nobs_dict contents:\n {obs_dict}")

    traj = dict(actions=[], obs=[], next_obs=[], sub_obs=[], uncertainties=[])
    
    for i in range(horizon):
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])
        sub_obs = copy.deepcopy(obs_dict["subtask_terms"])
       
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
            #print(f"found observation : {obs[ob]}")
        
        for subob in sub_obs:
            sub_obs[subob] = torch.squeeze(sub_obs[subob])

        # Check if environment image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        traj["obs"].append(obs)
        traj["sub_obs"].append(sub_obs)
        
        # Add dropout layers to the model and calculate uncertainty and remove at the end to not effect final action

        #hooks = inject_dropout_layers(policy=policy, probability=0.1)
        # uncertainty_dict = MC_dropout_uncertainty(policy=policy, obs=obs, niters=15)
        # traj['uncertainties'].append(uncertainty_dict['variance'])
        #remove_dropout_layers(hooks)

        # Compute actions
        actions = policy(obs)
      
        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        sub_obs = obs_dict["subtask_terms"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)


        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj
# tip over detection
def has_fallen(asset, threshold: float = -0.7, debug: bool = True) -> bool:
    """
    Detect if an IsaacLab asset has fallen over by checking its up vector.
    
    Args:
        asset: IsaacLab asset (e.g. robot, object) with .get_world_pose()
        threshold: minimum z-component of the up vector (1.0 = perfectly upright).
                   Typical cutoff ~0.7 (≈45° tilt).
        debug: if True, prints internal values for inspection.

    Returns:
        True if fallen, False otherwise.
    """
    quat = asset.data.body_link_quat_w   # (pos, quat), usually torch.Tensors
   # print("Beaker quat : ", quat)
    # Ensure torch tensor, shape (4,)
    if isinstance(quat, (list, tuple)):
        quat = torch.tensor(quat, dtype=torch.float32)
    quat = quat.view(-1, 4)  # shape (1,4)

    # Convert quaternion to wxyz ordering
    quat_wxyz = convert_quat(quat, to="wxyz")

    # Rotation matrix (1,3,3)
    R = matrix_from_quat(quat_wxyz)

    # World up vector is the third column of rotation matrix
    up = R[0, :, 2]   # shape (3,)
    uprightness = float(up[2])

    if debug:
        print(f"[has_fallen] quat={quat_wxyz.cpu().numpy().round(4)}")
        print(f"[has_fallen] up={up.cpu().numpy().round(4)} uprightness={uprightness:.3f}")

    return uprightness > threshold



def rollout_ensemble(ensemble, env, success_term, horizon, device, parameters,  use_recovery=True, rollout_num=0):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    ## do setup
    print("running rollout number : ", rollout_num)
    for policy in ensemble:
        policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[], sub_obs=[], 
                uncertainties=[], min_actions=[], max_actions=[],
                time_taken=[], joint_pos=[], recovery=[], failure=[], grasp=[], appr=[])
   
    ###### SETUP SWITCHING LOGIC ####
    switchingLogic = SwitchingLogic(parameters, horizon=horizon)
    certain_joint_positions = []

    ###### SET UP RECOVERY   ####
    use_recovery=True
    recovery_activated_during_rollout = 0
    print("rollout recovery enabled ? : ", use_recovery)
    
    ##### RECOVERY CONFIG  ####
    recovery_cooldown_duration = 300 # timesteps
    recovery_cooldown_timer = recovery_cooldown_duration
    recovery_cooldown_active = False
    recovery = []
    recovery_steps = 0

    ##### CONFIG RECOVERY CONTROLLER ####
    backup_controller  = BackupController(env, device)
    state_guess = 0
    last_state =0
    recovery_mode= True
    state_guessed = False
    recovery_triggered = False
    object_knocked = False

    print(f'Running for horizon {horizon}')
    for i in range(horizon):
        # Prepare observations
      
        obs = copy.deepcopy(obs_dict["policy"])
        sub_obs = copy.deepcopy(obs_dict["subtask_terms"])
       
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
            
        for subob in sub_obs:
            sub_obs[subob] = torch.squeeze(sub_obs[subob])
        traj["obs"].append(obs)
        traj["sub_obs"].append(sub_obs)
        
        # Check if environment image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        #### get policy action   ####
        metrics = ensemble_uncertainty(ensemble, obs)
        uncertainty = metrics['variance']
        policy_actions = metrics['mean']
        # fix state guess
        if sub_obs['grasp']:
                if state_guess <4 :
                    state_guess = 4
                    
        sm_actions , state_guess = backup_controller.get_controller_action(state_guess, 0)
        if last_state != state_guess:
            #print(f"State {state_guess.item()}")
            last_state = state_guess
            


        # this is the switching logic
        if switchingLogic.check(uncertainty, i):
                # uncertainty triggered
                if use_recovery and not recovery_cooldown_active:
                    # if we have switched to recovery
                    recovery_mode = True
                    recovery_activated_during_rollout+=1
                    if not recovery_triggered  :
                        print(f"[RECOVERY] Step {i}")
                        recovery_triggered = True
            
        if recovery_mode:
            # if holding the object, start with lifting 
            actions = sm_actions
      
        else:
            actions = policy_actions
            if recovery_cooldown_timer <= 0:
                recovery_cooldown_active = False
                recovery_cooldown_timer = recovery_cooldown_duration 
                print("Recovery Cooldown Ended")

            # Unnormalize actions
            if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
                actions = (
                    (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
                ) / 2 + args_cli.norm_factor_min
            actions = actions.to(device=device).view(1, env.action_space.shape[1])
        
        
        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        recovery.append(recovery_mode)
        # update obs from taking action
        obs = obs_dict["policy"]
        sub_obs = obs_dict["subtask_terms"]
        
        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)
        traj['uncertainties'].append(uncertainty)
        traj['max_actions'].append(metrics['max'])
        traj['min_actions'].append(metrics['min'])
        traj['time_taken'].append(metrics['time_taken'])
        traj['joint_pos'].append(obs['abs_joint_pos'])
        traj['recovery'].append(recovery)
        if torch.any(obs['object_knocked']):
            print("[MONITOR] object knocked over")
            object_knocked = True

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            traj['failure'].append(" ")
            grasp = torch.any(sub_obs['grasp'])
            traj['grasp'].append(grasp.item())
            appr = torch.any(sub_obs['appr_goal'])
            traj['appr'].append(appr.item())
            print(f"grasp : {traj['grasp']}, appr : {traj['appr']}")
            # print(f"grasp : {grasp.item()}, appr : {appr.item()}")
            return True, traj, recovery_activated_during_rollout, ""
        
        elif terminated or truncated:
            grasp = torch.any(sub_obs['grasp'])
            traj['grasp'].append(grasp.item())
            appr = torch.any(sub_obs['appr_goal'])
            traj['appr'].append(appr.item())
            #print(f"grasp : {traj['grasp']}, appr : {traj['appr']}")
            if i==horizon:
                #if the failure was caused by object collision
                traj['failure'].append("timeout")
                failure_reason = "timeout"
            else:
                #if failure due to timeout, record timeout
                traj['failure'].append("knocked")
                failure_reason = "object_knocked"
            return False, traj, recovery_activated_during_rollout, failure_reason
    #print("REcovery : ", traj["recovery"])
    traj['failure'].append("timeout")
    grasp = torch.any(sub_obs['grasp'])
    traj['grasp'].append(grasp.item())
    appr = torch.any(sub_obs['appr_goal'])
    traj['appr'].append(appr.item())
    return False, traj, recovery_activated_during_rollout, ""




def rollout_transformer(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment, supporting sequence-based models."""
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[], sub_obs=[], uncertainties=[])

    context_length = getattr(policy, "n_obs_steps", 1)  # works for transformer or rnn models
   # print(f"Policy's expected sequence length (policy.n_obs_steps): {getattr(policy, 'n_obs_steps', 1)}")

    obs_seq = []  # list of previous obs for context window

    # Initial obs_dict from env.reset()
   # print("Initial obs_dict shapes from env.reset():")
    # for k, v in obs_dict["policy"].items():
    #     print(f"  policy['{k}']: shape={v.shape}, ndim={v.ndim}")

    # try:
    #     if hasattr(policy.policy, 'encoder') and hasattr(policy.policy.encoder, 'input_obs_group_shapes'):
    #         print("Policy's internal expected input_obs_group_shapes:")
    #         for group, keys in policy.policy.encoder.input_obs_group_shapes.items():
    #             print(f"  Group '{group}':")
    #             for k, shape in keys.items():
    #                 print(f"    Key '{k}': shape={shape}, len(shape)={len(shape)}")
    #     else:
    #         print("Could not find policy.policy.encoder.input_obs_group_shapes.")
    # except Exception as e:
    #     print(f"Error accessing policy.policy.encoder.input_obs_group_shapes: {e}")


    for i in range(horizon):
        # Prepare current observations from env for policy input
        current_policy_obs = {}
        for k, v in obs_dict["policy"].items():
            # Ensure low-dim observations are 1D (D,)
            # If env returns (1, D), squeeze the batch dimension
            if v.ndim > 1 and v.shape[0] == 1:
                processed_v = v.squeeze(0).to(device) # Apply squeeze
                # print(f"  DEBUG: Squeezed '{k}' from {v.shape} to {processed_v.shape}") # Add debug print
            else:
                processed_v = v.to(device) # No squeeze needed, assume (D,)
            current_policy_obs[k] = processed_v

        # # AFTER processing, print the shapes in current_policy_obs
        # if i == 0: # Only print for the first step to avoid spamming
        #     print("current_policy_obs shapes AFTER squeezing (if applicable):")
        #     for k, v in current_policy_obs.items():
        #         print(f"  '{k}': shape={v.shape}, ndim={v.ndim}")


        # Handle image observations specifically (if any)
        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    image = obs_dict["policy"][image_name].to(device)
                    # Assuming image comes as (H, W, C) and needs to be (C, H, W)
                    # If it's already (1, H, W, C), squeeze the batch dim first.
                    if image.ndim == 4 and image.shape[0] == 1:
                        image = image.squeeze(0) # Remove initial batch if present

                    # Permute and normalize after ensuring no batch dim
                    image = image.permute(2, 0, 1).clone().float() # (C, H, W)
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    current_policy_obs[image_name] = image


        # Add subtask terms for logging/analysis, but they are not fed to the policy
        # as per your trace (only "policy" observations are listed in the policy's encoder).
        # You had prints for these before, assuming they are okay.

        # Append to context buffer. obs_seq should hold items with shape (D,) or (C, H, W)
        traj["obs"].append(current_policy_obs) # Store the current, processed observation
        obs_seq.append(current_policy_obs) # This is what forms the sequence input

        if len(obs_seq) > context_length:
            obs_seq = obs_seq[-context_length:]

        # Pad if not enough context
        if len(obs_seq) < context_length:
            padding_obs = {}
            for k, v in obs_seq[0].items(): # Use shape of first element in obs_seq for padding
                # If obs_seq[0][k] is (D,) then zeros_like creates (D,)
                # If obs_seq[0][k] is (C,H,W) then zeros_like creates (C,H,W)
                padding_obs[k] = torch.zeros_like(v)
            obs_seq = [padding_obs] * (context_length - len(obs_seq)) + obs_seq

        # Convert context list to batched sequence dict
        seq_input = {}
        for key in obs_seq[0]:
            # Each step[key] here should be (D,) or (C, H, W)
            # torch.stack will create (context_length, D) or (context_length, C, H, W)
            # unsqueeze(0) will add the batch dimension: (1, context_length, D) or (1, context_length, C, H, W)
            seq_input[key] = torch.stack([step[key] for step in obs_seq], dim=0).to(device=device)
            #seq_input[key] = torch.stack([step[key] for step in obs_seq], dim=0).unsqueeze(0).to(device=device)

        # Print the final seq_input shapes just before policy call
        # if i == 0: # Only print for the first step
        #     print("seq_input shapes for policy just before call:")
        #     for k, v in seq_input.items():
        #         print(f"  '{k}': shape={v.shape}, ndim={v.ndim}")

        # if i == 0: # Only print for the first step
        #     print("seq_input shapes for policy just before policy(seq_input) call (LAST CHECK):")
        #     for k, v in seq_input.items():
        #         print(f"  '{k}': shape={v.shape}, ndim={v.ndim}")

        # calculate uncertainty
        # uncertainty_dict = mc_dropout_uncertainty_eval(policy=policy, obs=seq_input, niters=15)
        # traj['uncertainties'].append(uncertainty_dict['variance'])

        # Compute action from sequence
        actions = policy(seq_input)

        #print(f"DEBUG (Pre-Unnorm): Policy output 'actions' shape: {actions.shape}, ndim: {actions.ndim}, type: {type(actions)}")

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min
        #print(f"DEBUG (Post-Unnorm): Policy output 'actions' shape: {actions.shape}, ndim: {actions.ndim}, type: {type(actions)}")
        # Convert policy output (torch.Tensor) to numpy array for env.step()
        # Assume actions comes as (1, Action_Dim) from policy, squeeze to (Action_Dim,)
        if isinstance(actions, torch.Tensor):
            # If it's a tensor, convert it to numpy, ensuring it's 1D (7,)
            # We already know policy returns (7,) so no squeeze is needed *here*.
            actions = actions.cpu().numpy()

        actions_tensor = torch.from_numpy(actions).to(device=device).float()
        actions_tensor = actions_tensor.unsqueeze(0) 
        # Apply actions
        #obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs_dict, _, terminated, truncated, _ = env.step(actions_tensor)
        # Record trajectory - traj["next_obs"] should append the raw observation dictionary from env.step().
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs_dict["policy"])


        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj



def load_ensemble(device, ensemble_path):
    with open(ensemble_path, 'r') as file:
        ensemble_paths = [path.strip() for path in file.readlines()]
    ensemble = []
    for path in ensemble_paths:
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=path, device=device, verbose=True)
        ensemble.append(policy)
    print(f"[DEBUG] loaded {len(ensemble)} policies")
    return ensemble

def clear_files(*paths : str):
    if len(paths) < 1:
        raise ValueError("At least one path is required.")
    if os.path.exists(paths[0]):
        #overwrite = input(f"You are about to overwrite rollout data at: {paths}\nContinue (y/n):")
        overwrite = "y"
        match overwrite.lower():
            case 'y' | 'Y':
                for path in paths:
                    with open(path, 'w') as file:
                        pass
            case _:
                exit()
            
                

def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    #create a log handler 
   # loghelper = LoggingHelper()

    

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Get timeout signal 
    timeout = env_cfg.terminations.time_out
    #env_cfg.terminations.time_out = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # load the stack cube ensemble
    # stack_cube_ensemble = load_ensembl*--------------e(device, ensemble_path='scripts/imitation_learning/robomimic/stack_cube_ensemble.txt')    
    
    #pick_place_ensemble = load_ensemble(device, ensemble_path='scripts/imitation_learning/robomimic/ensembles.txt')
    pick_place_ensemble = load_ensemble(device, ensemble_path='scripts/imitation_learning/robomimic/med_ensembles.txt')
    # pick_place_ensemble_30 = load_ensemble(device, ensemble_path='scripts/imitation_learning/robomimic/pick_place_ensemble_30_paths.txt')
 
    # Lets set these to the 0.99 confidence
    parameters = {
        'beaker_lift' :{
            0 : {
                "confidence_level": 0.010360572377319478,
                "window_size": 9,
                "max_peaks": 9
            },
            1 : {
                "confidence_level": 0.012117819935441015,
                "window_size": 9,
                "max_peaks": 9
            },
            2 : {
                "confidence_level": 0.0114854090040122,
                "window_size": 9,
                "max_peaks": 9
            },
            3 : {
                "confidence_level": 0.01721086513038985,
                "window_size": 9,
                "max_peaks": 9
            },
            4 : {
                "confidence_level": 0.01516669949421836,
                "window_size": 9,
                "max_peaks": 9
            },
            5 : {
                "confidence_level": 0.011445348395548671,
                "window_size": 9,
                "max_peaks": 9
            },
            6 : {
                "confidence_level": 0.0001330413469565007,
                "window_size": 15,
                "max_peaks": 9
            },

        },
    }

    model_arch = "BC_RNN_GMM"
    task = "pick_place" # stack_cube or pick_place
    model_name = f"ensemble"
    number = 'no_recovery_video'

    results_path = f"docs/training_data/{task}/uncertainty_rollout_{task}/{model_name}/run_{number}"

    uncertainties_path = f"{results_path}/uncertainties{number}.txt"
    actions_path = f"{results_path}/actions{number}.txt"
    min_actions_path = f"{results_path}/min_actions{number}.txt"
    max_actions_path = f"{results_path}/max_actions{number}.txt"
    time_taken_path = f"{results_path}/time_taken{number}.txt"
    recovery_activated_during_rollout_path = f"{results_path}/recovery_activated_during_rollout.txt"
    rollout_uncert_path = f"{uncertainties_path}/rollout_uncerts"
    rollout_log_path = f"{uncertainties_path[:len(uncertainties_path)-4]}_rollout_log.txt"

    rollout_log_path = f"{uncertainties_path[:len(uncertainties_path)-4]}_rollout_log.txt" # remove the '.txt' and add rollout_log.txt

    uncertainty_results_path = f"{results_path}/uncertainty_plots"
    trajectory_results_path = f"{results_path}/trajectory_plots"
    os.makedirs(uncertainty_results_path, exist_ok=True)
    os.makedirs(trajectory_results_path, exist_ok=True)


    # clear files
    clear_files(
        uncertainties_path, actions_path, 
        min_actions_path, max_actions_path, 
        time_taken_path, recovery_activated_during_rollout_path
    )
    
    # Run policy
    results = []
    failed_no_intervention =[]
    intervention = []
    intervention_success = []
    intervention_failure = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        terminated, traj, recovery_activated_during_rollout, failure = rollout_ensemble(pick_place_ensemble[:args_cli.ensemble_size], env, success_term, args_cli.horizon, device, parameters['beaker_lift'], use_recovery=args_cli.use_recovery, rollout_num=trial)
        # save the uncertainties
        print("Finished rollout, recovery needed : ", recovery_activated_during_rollout)
        #print("actions shape : ", traj['actions'])
        
        if not terminated:
            print(f"failure : {traj['failure']}")
        for j in range(7):
            with open(f"docs/rollouts/rollout_logging_no_recovery_joint{j}.txt", "a") as file:
                #file.write(f"Logs for joint {j}\n")
                for i, var in enumerate(traj['actions']):
                    # get all the info per joint
                    act = str(traj['actions'][i][0][j])
                    max = str(traj['max_actions'][i].values.tolist()[j])
                    min = str(traj['min_actions'][i].values.tolist()[j])
                    uncert = str(traj['uncertainties'][i].tolist()[j])
                    #print("joint pos : ", str(traj['joint_pos'][i][0].tolist()[j]))
                    joint_pos = str(traj['joint_pos'][i][0].tolist()[j])
                    recovery = "False"#str(traj['recovery'][0][i])
                   # print("writing recovery : ", recovery)
                    line = f"{i} : {act} : {max} : {min} : {uncert} : {joint_pos}: {recovery}: {terminated}\n"
                    file.write(line)
        # this logs the outcome of the rollout - the failure methods
        for j in range(7):
            joint_uncert = [t[j].item() for t in traj['uncertainties']]
            with open(f"docs/rollouts/rollout_uncerts_joint{j}.txt", "a") as f:
                line = ",".join(["{:.10f}".format(v) for v in joint_uncert])
                f.write(line + f" : {terminated} \n")
       # for i in range(len(traj['failure'])):
        with open(f"docs/rollouts/rollout_outcomes.txt", "a") as file:
            if terminated:
               # print(f"grasp : {traj['grasp']}, appr : {traj['appr']}")
                #trialnum : outcome  : recovery needed ? : failure mode : grasp subtask : appr subtask
                line=f"{trial}: {terminated} : {recovery_activated_during_rollout} : : {True} : {True} \n"
            else : 
                grasp = traj['grasp'][0]
                appr = traj['appr'][0]
                print(f"Subatask : {grasp} , {appr}")
                line=f"{trial} : {terminated} : {recovery_activated_during_rollout} : {traj['failure'][0]} : {grasp} : {appr}\n" 
               # print(f"grasp : {traj['grasp'][0]}, appr : {traj['appr'][0]}")
            file.write(line)

        
        results.append(terminated)
        if not terminated:
            if recovery_activated_during_rollout <=0:
                failed_no_intervention.append(1)

        if recovery_activated_during_rollout > 0:
            intervention.append(1)
            if terminated:
                intervention_success.append(1)
            else:
                intervention_failure.append(1)
        else:
            intervention.append(0)
        print(f'Current rate {results.count(True)/len(results)}')
        print(f"[INFO] Trial {trial}: {terminated}\n")
        #loghelper.stopEpoch(trial)

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f'Failed runs with no intervention : {sum(failed_no_intervention)}')
    print(f"Intervention rate : {sum(intervention)/len(intervention)}. Of which success : {sum(intervention_success)}, of which failed {sum(intervention_failure)}")
    
    print(f"Trial Results: {results}\n")


    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()