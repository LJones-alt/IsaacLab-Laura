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
#parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
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
parser.add_argument("--use_joint_pos", action="store_true", default=False, help="Use Joint Position State Machine Controller.")
parser.add_argument("--gripper_orient", type=int, default=0)


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
from test_controller import TestController
from test_controller_top import TestController as TopTestController
from test_joint_pos_controller import TestJointPosControllerSM

from isaacsim.util.debug_draw import _debug_draw
from isaaclab.safety.apf import APF , Hazard
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.forwards_kinematics import ForwardsDynamics
from isaaclab.safety.safety_barrier import JP_APF


def draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max, colour=(1.0, 1.0, 0.0, 1.0)):
    """Draws a wireframe box representing the APF boundaries."""
    # Define the 8 corners of the box
    corners = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ]
    
    # Define the 12 edges (pairs of corner indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom square
        (4, 5), (5, 6), (6, 7), (7, 4), # Top square
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
    ]
    
    point_list_1 = []
    point_list_2 = []
    for start, end in edges:
        point_list_1.append(corners[start])
        point_list_2.append(corners[end])
        
    # Draw lines (Color is RGBA: red)
    colors = [colour] * len(point_list_1)
    sizes = [2.0] * len(point_list_1)
    
    draw_interface.draw_lines(point_list_1, point_list_2, colors, sizes)


def rollout_ensemble(env, success_term, horizon, device, use_joint_pos: bool = False):
    """Perform a single rollout of the policy in the environment.
    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.
        use_joint_pos: Whether to use Joint Position state machine controller.
    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    x_min=-0.2
    x_max=0.40
    y_min=-0.73
    y_max=0.73 
    z_min=0.0 
    z_max=0.9
    env.reset()
    draw_interface = _debug_draw.acquire_debug_draw_interface()
    ###### SETUP SWITCHING LOGIC ####
    test_obst = [1.0, 1.0, 1.0, 0.0]
    joint_apf=JP_APF(device=torch.device("cuda:0"),obst=test_obst)
    joint_apf.model.eval()
    if use_joint_pos:
        print(f"Joint pos controller")
        test_controller = TestJointPosControllerSM(device, env)
    else:
        print(f"Diff IK controller")
        #test_controller = TestController(device, env, args_cli.gripper_orient)
        test_controller=TopTestController(device, env, args_cli.gripper_orient )
    #test_controller = TestController(device, env, args_cli.gripper_orient)
    for i in range(horizon):
        action = test_controller.get_action()
        robot = env.unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos[0, 0:7]
        obs_dict = env.observation_manager.compute_group("policy")
        q_dot = obs_dict["joint_vel"] # maybe robot.data.joint_vel ? 
        safety_val_t0 , joint_vels_t0=joint_apf.get_joint_vals(joint_pos, q_dot, 0.1)
        #print(f"Action: {action}")
        obs_dict, _, terminated, truncated, _ = env.step(action)
        if safety_val_t0 > 0.01:
            wall_color = (0.0, 1.0, 0.0, 1.0)  # Green
        elif safety_val_t0 >= 0.0:
            wall_color = (1.0, 1.0, 0.0, 1.0)  # Yellow
        else:
            wall_color = (1.0, 0.0, 0.0, 1.0)  # Red
        draw_interface.clear_lines()
        draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max, colour=wall_color)

        # Check success condition
        if bool(success_term.func(env, **success_term.params)[0]):
            test_controller.reset()
            return True
        # Check termination or truncation condition
        elif terminated or truncated:
            test_controller.reset()
            return False
    test_controller.reset()
    return False
    

       
    
          
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

    # setup APF
    # Define walls
    x_min=-0.2
    x_max=0.40
    y_min=-0.73
    y_max=0.73 
    z_min=0.0 
    z_max=0.9
    draw_interface = _debug_draw.acquire_debug_draw_interface()
    draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max )

    apf_name = "no_obs_top_down"
    retrain = False
    test_obst = [1.0, 1.0, 1.0, 0.0]
    

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        rollout_ensemble(env, success_term, args_cli.horizon, device, args_cli.use_joint_pos)

        # save the uncertainties
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()