# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments and APF safety."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments with APF.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--task", type=str, default="Dev-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--real_robot", type=bool, default=False, help="Use real robot.")
parser.add_argument("--retrain", type=bool, default=False, help="Retrain APF on new sim")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--apf_model",
    type=str,
    default="docs/apf/safety_wall_model.pth",
    help="Path to the pre-trained APF model.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import omni.log
from isaacsim.util.debug_draw import _debug_draw

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.safety.apf import APF , Hazard

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg
if args_cli.real_robot:
    from isaaclab.sim2real.bridgeclient import BridgeClient
import time


def draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max):
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
    colors = [[1.0, 0.0, 0.0, 1.0]] * len(point_list_1)
    sizes = [2.0] * len(point_list_1)
    
    draw_interface.draw_lines(point_list_1, point_list_2, colors, sizes)


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment and APF safety.
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task or "Cube" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        if hasattr(env_cfg.commands, "object_pose"):
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        # env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # get obstacle positions from environment
    hazards = [
        Hazard(x=0.2, y=0.2, z=0.2, r=0.15, name='obstacle1'),
        Hazard(x=-0.2, y=-0.05, z=0.2, r=0.05, name='obstacle2')
    ]
    # no hazards
    hazards = [
        Hazard(x=0, y=0, z=0, r=0, name='obstacle1')
    ]

    # Define walls
    x_min=-0.2
    x_max=0.40
    y_min=-0.73
    y_max=0.73 
    z_min=0.0 
    z_max=0.9

    apf_name = "no_obs_top_down"
    retrain = args_cli.retrain
    # Initialize APF
    
    apf = APF(hazards, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max, plot_graph=True, name=apf_name , retrain=retrain)

    # Load the trained model
    if os.path.exists(args_cli.apf_model):
        apf.load_model(args_cli.apf_model)
    else:
        omni.log.error(f"APF model not found at {args_cli.apf_model}")

    # Initialize debug draw interface and draw walls
    draw_interface = _debug_draw.acquire_debug_draw_interface()
    draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max )

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # Create teleop device
    sensitivity = args_cli.sensitivity
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.005 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
        )
    else:
        omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
        env.close()
        simulation_app.close()
        return

    # Add callbacks
    for key, callback in teleoperation_callbacks.items():
        try:
            teleop_interface.add_callback(key, callback)
        except Exception as e:
            omni.log.warn(f"Failed to add callback for key {key}: {e}")

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")
    if args_cli.real_robot:
        bridge = BridgeClient()
    prev_gripper_action = None
    # simulate environment
    while simulation_app.is_running():
        try:
            with torch.no_grad():
                # get device command
                action = teleop_interface.advance()

                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1).to(env.device)
                  #print(f"Teleop actions : {actions}")
                    # Apply APF Safety
                    # Get current EE position
                    obs_dict = env.observation_manager.compute_group("policy")
                    eef_pos = obs_dict["eef_pos"] # (num_envs, 3)
                    eef_quat= obs_dict["eef_quat"] # (num_envs, 4)
                    # Modify the position delta (first 3 components of action)
                    for i in range(env.num_envs):
                        nominal_velocity = actions[i, 0:3]
                        current_pos = eef_pos[i]
                        
                        # We use torch.enable_grad() because APF needs gradients for repulsion
                        with torch.enable_grad():
                            safe_velocity = apf.compute_safe_velocity(
                                current_pos, 
                                nominal_velocity, 
                                safety_margin=0.05, 
                                repulsion_gain=0.1
                            )
                        #print(f"Safe velocity: {safe_velocity}")
                        #print(f"Nominal velocity: {nominal_velocity}")
                        # applies APF safety constraints
                        actions[i, 0:3] = safe_velocity

                        # for ros2 brideg : 
                        abs_pos = eef_pos + actions[:, 0:3]
                        abs_quat_ros = torch.cat([eef_quat[:, 1:], eef_quat[:, 0:1]], dim=-1)
                        abs_pose_bridge = torch.cat([abs_pos, abs_quat_ros], dim=-1)
                        if args_cli.real_robot:
                            if bridge is not None:
                            # only send new pose every 5
                        
                        #   print(f"Trying to publish...")
                            # Send absolute pose to bridge
                           # bridge.publish_command(abs_pose_bridge[0])
                            # Publish Joint States for MoveIt 2
                                bridge.publish_joints(obs_dict['abs_joint_pos'][0])
                        #  print(f"Published joint states: {obs['abs_joint_pos'][0]}")
                        
                            # Handle gripper toggling
                            
                            # print(f"Grippper pos  : {curr_gripper}, left {curr_gripper[0,0]}")
                            # print(f"Gripper action: {actions[0,-1]}")
                            if actions[0,-1] >0:
                                # if action is open
                                gripper_action = 1
                                if gripper_action != prev_gripper_action:
                                    print(f"DEBUG: Triggering Open Gripper ")
                                    bridge.open_gripper()
                            else:
                                gripper_action =-1
                                if gripper_action!=prev_gripper_action:
                                    print(f"DEBUG: Triggering Close Gripper ")
                                    bridge.close_gripper()   
                            prev_gripper_action = gripper_action
                    # apply actions
                    env.step(actions)
                    if args_cli.real_robot:
                        # only sleep if we are really sending on ROS2
                        time.sleep(0.02)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
