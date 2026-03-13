# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--task", type=str, default="Dev-IK-Rel-Place-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
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

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

# --- UDP Bridge imports ---
import socket
import json
import time
import numpy as np

class BridgeClient:
    def __init__(self, port=5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ('127.0.0.1', port)
        self.sock.settimeout(0.5)
        self._last_print_time = 0

    def publish_command(self, actions):
        """Publish 7-DOF Cartesian target: [x, y, z, qx, qy, qz, qw]"""
        if hasattr(actions, 'cpu'):
            act_list = actions.cpu().numpy().flatten().tolist()
        else:
            act_list = np.array(actions).flatten().tolist()
        
        if len(act_list) < 7: return
        
        payload = {'type': 'cartesian', 'data': act_list[:7]}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)
        
        # Throttled debug print (every 1 second)
        curr_time = time.time()
        if curr_time - self._last_print_time > 1.0:
            print(f"DEBUG: Sent Cartesian to Bridge: {act_list}")
            self._last_print_time = curr_time

    def publish_joints(self, positions):
        """Publish 7-DOF Joint positions"""
        if hasattr(positions, 'cpu'):
            pos_list = positions.cpu().numpy().flatten().tolist()
        else:
            pos_list = np.array(positions).flatten().tolist()
        
        if len(pos_list) < 7: return
        
        payload = {'type': 'joint', 'data': pos_list[:7]}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)

        # Throttled debug print (every 1s)
        curr_time = time.time()
        if curr_time - self._last_print_time > 1.0:
            print(f"DEBUG: Sent Joint Target to Bridge: {[round(p, 3) for p in pos_list[:7]]}")
            self._last_print_time = curr_time

    def open_gripper(self):
        print("DEBUG: Sending Open Gripper to Bridge")
        payload = {'type': 'gripper', 'width': 0.08}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)

    def close_gripper(self):
        print("DEBUG: Sending Close Gripper to Bridge")
        payload = {'type': 'gripper', 'width': 0.0}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)

    def get_robot_state(self):
        payload = {'type': 'query'}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)
        try:
            data, _ = self.sock.recvfrom(2048)
            return json.loads(data.decode())
        except socket.timeout:
            return None

    def init_real(self, sim_pose):
        print("[INFO] Initializing real robot synchronization via bridge...")
        self.publish_command(sim_pose)
        target_pos = np.array(sim_pose[:3])
        
        while True:
            state = self.get_robot_state()
            if state and state.get('real_pose'):
                real_pos = np.array(state['real_pose']['pos'])
                dist = np.linalg.norm(target_pos - real_pos)
                print(f"[INFO] Distance to target: {dist:.4f}m")
                if dist < 0.05:
                    print("[INFO] Real robot synchronized with simulation!")
                    break
            else:
                print("[INFO] Waiting for bridge on 127.0.0.1:5005...")
            time.sleep(0.5)


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # Initialize UDP Bridge Client
    bridge = BridgeClient()

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        # External cameras are not supported with XR teleop
        # Check for any camera configs and disable them
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            omni.log.warn(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
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
                omni.log.error("Supported devices: keyboard, spacemouse, gamepad, handtracking")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    omni.log.warn(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        omni.log.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()
    prev_gripper_action = None

    # Sync real robot with simulation initial pose
    if bridge is not None:
        # Get simulated robot initial EE pose
        try:
            # We use the observation manager to get the tracked end-effector pose
            # Based on your environment info, these keys are available in the 'policy' group
            obs_dict = env.observation_manager.compute_group("policy")
            
            # eef_pos is (num_envs, 3), eef_quat is (num_envs, 4)
            eef_pos = obs_dict["eef_pos"][0].cpu().numpy()
            eef_quat = obs_dict["eef_quat"][0].cpu().numpy() # [qw, qx, qy, qz]
            
            # ROS expects [qx, qy, qz, qw], Isaac uses [qw, qx, qy, qz]
            sim_init_pose = [
                float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
                float(eef_quat[1]), float(eef_quat[2]), float(eef_quat[3]), float(eef_quat[0])
            ]
            sim_pose = [0.5208, 0.0096, 0.3751, 0.0360, 0.7414, -0.0706, 0.6663]
            print(f"Simulated robot initial pose: {sim_init_pose}")
            # bridge.init_real(sim_init_pose)
        except Exception as e:
            print(f"[WARN] Could not retrieve sim initial pose from observations: {e}")
            print("[INFO] Skipping real robot synchronization.")

    print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1).to(env.device)

                    # Compute absolute pose for the ROS bridge
                    with torch.no_grad():
                        # Get latest observations from the 'policy' group
                        obs_dict = env.observation_manager.compute_group("policy")
                        eef_pos = obs_dict["eef_pos"] # (num_envs, 3)
                        eef_quat = obs_dict["eef_quat"] # (num_envs, 4) -> [qw, qx, qy, qz]

                        # Calculate absolute target for bridge (current + delta)
                        # actions[:, 0:3] is the position delta
                        abs_pos = eef_pos + actions[:, 0:3]
                        
                        # For orientation, we map Isaac [qw, qx, qy, qz] to ROS [qx, qy, qz, qw]
                        # and keep the current orientation (or integrate if needed, but simple is better)
                        abs_quat_ros = torch.cat([eef_quat[:, 1:], eef_quat[:, 0:1]], dim=-1)
                        
                        # Construct a 7-DOF absolute pose [x, y, z, qx, qy, qz, qw]
                        abs_pose_bridge = torch.cat([abs_pos, abs_quat_ros], dim=-1)
                        print(f"absolute pose: {eef_pos}, {eef_quat}")
                        print(f"absolute pose action: {abs_pose_bridge}")
                    # ROS 2 Synchronization via Bridge
                    if bridge is not None:
                        # Send absolute pose to bridge
                        bridge.publish_command(abs_pose_bridge[0])
                        
                        # Send joint positions for MoveIt 2
                        obs_dict_full = env.observation_manager.compute_group("policy")
                        if "joint_pos" in obs_dict_full:
                            bridge.publish_joints(obs_dict_full["joint_pos"][0])

                        # Handle gripper toggling
                        curr_gripper = float(actions[0, -1])
                        if prev_gripper_action is not None:
                            if curr_gripper != prev_gripper_action:
                                if curr_gripper > 0:
                                    bridge.open_gripper()
                                else:
                                    bridge.close_gripper()
                        prev_gripper_action = curr_gripper

                    # apply actions (relative as intended by the environment)
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
        
    env.close()
    print("Environment closed")

def process_action(action):
    ## handler that will spit out ROS2 commands 

    print(f"Teleop Actions : {action}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
