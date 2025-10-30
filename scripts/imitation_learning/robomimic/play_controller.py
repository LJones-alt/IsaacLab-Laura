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
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

#import chills.tasks
#from chills.tasks.mdp import upright_tilt_deg

def rollout_ensemble(env, success_term, horizon, device):
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
    
    
    obs_dict, _ = env.reset()
    #print(f"obs_dict type: {type(obs_dict)},\nobs_dict contents:\n {obs_dict}")

    traj = dict(actions=[], obs=[], next_obs=[], sub_obs=[], 
                uncertainties=[], min_actions=[], max_actions=[],
                time_taken=[], joint_pos=[], recovery=[], failure=[], grasp=[], appr=[])
   
    ###### SETUP SWITCHING LOGIC ####

    sim = env.unwrapped.sim
    num_envs = env.num_envs


    certain_joint_positions = []

    ### SET USE RECOVERY TO FALSE   ####
    
    
    # Set up recovery controller 
    robot = env.unwrapped.scene["robot"]
    print(f"Unwrapped sim : {sim}")
    ee_recovery_pos = torch.tensor([[0.5206, 0.0096, 0.3751]], device=device)

    ee_recovery_rot = torch.tensor([[ 0.6664,  0.0360,  0.7414, -0.0705]], device=device)
    # lets change this into the 6 element tensor they are expecting 
    roll,pitch,yaw = euler_xyz_from_quat(ee_recovery_rot)
    ee_recovery_goal = torch.cat([ee_recovery_pos, ee_recovery_rot], dim =-1)
    print(f"I got this roll {roll}, pitch {pitch}, yaw {yaw} and full tensor {torch.cat([roll, pitch, yaw], dim=-1)}\n i need to add with {ee_recovery_pos}")
    ee_rpy = torch.cat([roll, pitch, yaw], dim=-1)
    ee_rpy = ee_rpy.unsqueeze(0)
    ee_goal = torch.cat([ee_recovery_pos, ee_rpy], dim=1)
    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.unwrapped.scene.num_envs, device=sim.device)
    
    
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(env.unwrapped.scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    print(f'Running for horizon {horizon}')
    for i in range(horizon):
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # current EE pose in base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        print(f"ee_pos_b {ee_pos_b}, ee_quat_b : {ee_quat_b}")

        # target EE pose (absolute, world frame → base frame)
        target_pos_w = ee_recovery_pos     # [1,3]
        target_quat_w = ee_recovery_rot    # [1,4]
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos_w, target_quat_w
        )
        print(f"target_pos_b {target_pos_b}, target_quat_b {target_quat_b}")

        # --- compute relative pose command (6D) ---
        # translation delta
        delta_pos = target_pos_b - ee_pos_b    # [1,3]
        print(f"Delta pos {delta_pos}")
        # rotation delta (ΔR = R_target * R_currentᵀ)
        delta_rot = target_quat_b - ee_quat_b
        print(f"Delat Rot {delta_rot}")
        # convert ΔR to euler angles (XYZ convention)
        r,p,y = euler_xyz_from_quat(delta_rot, wrap_to_2pi=False)
        print(f"[DEBUG] Delat R {r}, P {p}, Y {y}")
        delta_rpy = torch.cat([r,p,y], dim=-1)
        delta_rpy = torch.unsqueeze(delta_rpy,0)
        print(f"[DEBUG] RPY {delta_rpy}")
        # --- build relative command ---
        ee_goal = torch.cat([delta_pos, delta_rpy], dim=-1)  # [1,6]

        
        # debug info
        print(f"[DEBUG] ee_goal (Δx,Δy,Δz,Δr,Δp,Δy): {ee_goal}")
        print(f"[DEBUG] Current pos_b: {ee_pos_b}")
        print(f"[DEBUG] Target pos_b: {target_pos_b}")

        # --- send command to controller ---
        diff_ik_controller.set_command(ee_goal, ee_pos_b, ee_quat_b)

        # compute joint targets
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # (optional) denormalize actions if necessary
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = ((actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)) / 2 + args_cli.norm_factor_min

        # --- execute recovery movement ---
        pos_err = torch.norm(target_pos_b - ee_pos_b)
        actions=joint_pos_des
        metrics = ensemble_uncertainty(ensemble, obs)
        metrics['min']=actions
        metrics['max']=actions
        uncertainty = 0

        if pos_err > 1e-3:
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            env.scene.write_data_to_sim()
            obs_dict, _, terminated, truncated, _ = env.step(joint_pos_des)
        else:
            print(f"[Recovery] Target reached — ending recovery mode.")
            recovery_mode = False
            recovery_cooldown_active = True
            print("[Recovery] Cooldown Activated.")






                

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
    arm_action = env_cfg.actions.arm_action
    # Get timeout signal 
    timeout = env_cfg.terminations.time_out
    #env_cfg.terminations.time_out = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed9
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
   
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        terminated, traj, recovery_activated_during_rollout = rollout_ensemble( env, success_term, args_cli.horizon, device)
        # save the uncertainties
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()