# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments and APF safety."""

"""Launch Isaac Sim Simulator first."""

from numpy.lib import nanfunctions
from enum import Enum
from PIL import ImageSequence
from six.moves import urllib_robotparser
import argparse
from collections.abc import Callable
import os
import sys

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
from isaaclab.utils.forwards_kinematics import ForwardsDynamics
from isaaclab.safety.safety_barrier import JP_APF
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


class Status(Enum):
    OK = 0,
    RISK = 1,
    RECOVERY = 2


def draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max, colour=(1.0, 1.0, 0.0, 1.0)):
    """Draws a wireframe box representing the APF boundaries."""
    corners = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    point_list_1 = []
    point_list_2 = []
    for start, end in edges:
        point_list_1.append(corners[start])
        point_list_2.append(corners[end])
        
    colors = [colour] * len(point_list_1)
    sizes = [2.0] * len(point_list_1)
    draw_interface.draw_lines(point_list_1, point_list_2, colors, sizes)


def get_desired_vel(env, actions, robot, ik_controller):
    current_q = robot.data.joint_pos[:, 0:7]
    hand_body_idx = robot.find_bodies("panda_hand")[0]
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    
    ee_body_idx = ee_frame.body_physx_index if hasattr(ee_frame, "body_physx_index") else 7
    jacobian = robot.root_physx_view.get_jacobians()[:, hand_body_idx, :, 0:7]
    delta_pose = actions[:, 0:6]
    ik_controller.set_command(delta_pose, ee_pos=ee_pos, ee_quat=ee_quat)
    q_target = ik_controller.compute(
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        jacobian=jacobian,
        joint_pos=current_q,
    )

    dt = env.physics_dt
    commanded_joint_vel = (q_target - current_q) / dt
    commanded_joint_vel = torch.where(
        torch.abs(commanded_joint_vel) < 1e-5,
        torch.tensor(0.0, device=commanded_joint_vel.device, dtype=commanded_joint_vel.dtype),
        commanded_joint_vel
    )
    QDOT_MAX = torch.tensor([1.175, 1.175, 1.175, 1.175, 1.610, 2.610, 2.610], device=env.device)
    q_dot_des = torch.clamp(commanded_joint_vel, -QDOT_MAX, QDOT_MAX)
    
    return q_dot_des, q_target


def print_live_dashboard(data, first_run=[True]):
    """Overwrites a fixed multi-line block in the terminal using ANSI escape codes."""
    lines = [
        "==================== ROBOT TELEOP & SAFETY DASHBOARD ====================",
        f" [Teleop Actions]   : {data['action'].detach().cpu().numpy().round(4)}",
        f" [Desired q]        : {data['q_des'].detach().cpu().numpy().round(4)}",
        f" [Robot Joint Pos]  : {data['joint_pos'].detach().cpu().numpy().round(4)}",
        f" [Safety Status]    : {data['status'].name} (Val: {data['safety_val']:.4f})",
        f" [Desired q_dot]    : {data['q_dot'].detach().cpu().numpy().round(4)}",
        f" [Command Modifier] : {data['command_mod'].detach().cpu().numpy().round(4)}",
        f" [Final Command]    : {data['command'].detach().cpu().numpy().round(4)}",
        "========================================================================="
    ]
    output = "\n".join(lines)
    num_lines = len(lines)
    
    if not first_run[0]:
        sys.stdout.write(f"\033[{num_lines}A\033[J")
    else:
        first_run[0] = False
        
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task or "Cube" in args_cli.task:
        if hasattr(env_cfg.commands, "object_pose"):
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=True, 
        ik_method="dls"
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
    print(f"[INFO] Made Controller ")

    hazards = [Hazard(x=0, y=0, z=0, r=0, name='obstacle1')]

    x_min=-0.2
    x_max=0.40
    y_min=-0.73
    y_max=0.73 
    z_min=0.0 
    z_max=0.9

    apf_name = "no_obs_top_down"
    retrain = args_cli.retrain
    apf = APF(hazards, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max, plot_graph=True, name=apf_name, retrain=retrain)
    test_obst = [0.0, 0.0, 0.0, 0.0]
    joint_apf = JP_APF(device=torch.device("cuda:0"), obst=test_obst)
    joint_apf.model.eval()    
    
    field_influence = 0.1
    robot_state = Status.OK

    draw_interface = _debug_draw.acquire_debug_draw_interface()
    draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max)

    should_reset_recording_instance = False
    teleoperation_active = True

    def reset_recording_instance() -> None:
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("\nReset triggered")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("\nTeleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("\nTeleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

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

    for key, callback in teleoperation_callbacks.items():
        try:
            teleop_interface.add_callback(key, callback)
        except Exception as e:
            omni.log.warn(f"Failed to add callback for key {key}: {e}")

    print(f"Using teleop device: {teleop_interface}")

    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.\n")
    if args_cli.real_robot:
        bridge = BridgeClient()
    prev_gripper_action = None
    safety_val = 0.1
    is_in_recovery = False
    last_recovery_state = False
    held_q_target = None
    robot = env.unwrapped.scene["robot"]

    while simulation_app.is_running():
        try:
            with torch.enable_grad():
                action = teleop_interface.advance()
                actions = action.repeat(env.num_envs, 1).to(env.device)
                teleop_active_input = torch.any(torch.abs(actions[:, 0:6]) > 1e-4)
                
                if teleop_active_input:
                    q_dot_des, q_des = get_desired_vel(env, actions, robot, ik_controller)
                    held_q_target = q_des
                else:
                    q_dot_des = torch.zeros((env.num_envs, 7), device=env.device)
                    if held_q_target is None:
                        held_q_target = robot.data.joint_pos[:, 0:7].clone() 
                    q_des = held_q_target

                joint_pos = robot.data.joint_pos[0, 0:7]
                joint_vels = robot.data.joint_vel[0, 0:7]
                
                safety_val_t0, joint_vels_t0 = joint_apf.get_joint_vals(joint_pos, q_dot_des, 0.009)
                safety_val = safety_val_t0[0].cpu().squeeze().tolist()
                
                if safety_val >= 0.001:
                    robot_state = Status.OK
                    wall_color = (0.0, 1.0, 0.0, 1.0)
                elif safety_val >= -0.1:
                    robot_state = Status.RISK
                    wall_color = (1.0, 1.0, 0.0, 1.0)
                else:
                    robot_state = Status.RECOVERY
                    wall_color = (1.0, 0.0, 0.0, 1.0)

                draw_interface.clear_lines()
                draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max, colour=wall_color)
                obs_dict = env.observation_manager.compute_group("policy")
                
                if args_cli.real_robot:
                    if bridge is not None:
                        bridge.publish_joints(obs_dict['abs_joint_pos'][0])
                    if actions[0, -1] > 0:
                        gripper_action = 1
                        if gripper_action != prev_gripper_action:
                            bridge.open_gripper()
                    else:
                        gripper_action = -1
                        if gripper_action != prev_gripper_action:
                            bridge.close_gripper()   
                    prev_gripper_action = gripper_action

                # Match robot state to establish final command
                command = torch.zeros((7,), device=env.device)
                match robot_state:
                    case Status.OK:
                        command = q_dot_des[0]
                        command_modifier = torch.zeros((7,), device=env.device)
                    case Status.RISK:
                        grad = joint_vels_t0[0]
                        grad_norm = grad.norm()
                        grad_dir = grad / grad_norm if grad_norm > 1e-6 else torch.zeros_like(grad)
                        field_nudge = field_influence * grad_dir
                        command_modifier = field_nudge
                        command = q_dot_des[0] + field_nudge
                        QDOT_MAX = torch.tensor([1.175, 1.175, 1.175, 1.175, 1.610, 2.610, 2.610], device=env.device)
                        command = torch.clamp(command, -QDOT_MAX, QDOT_MAX)
                        #safety_val_t0, joint_vels_t0 = joint_apf.get_joint_vals(joint_pos, command, 0.009)
                        #command = torch.zeros((7,), device=env.device)
                    case Status.RECOVERY:
                        command = torch.zeros((7,), device=env.device)
                        command_modifier = torch.zeros((7,), device=env.device)

                # Assemble and print dashboard state block
                dashboard_data = {
                    "action": actions[0],
                    "q_des": q_des[0],
                    "q_dot": q_dot_des[0],
                    "joint_pos": joint_pos,
                    "status": robot_state,
                    "safety_val": safety_val,
                    "command": command,
                    "command_mod" : command_modifier
                }
                print_live_dashboard(dashboard_data)

                # Send commands to sim
                env_actions = torch.zeros((env.num_envs, 8), device=env.device)
                env_actions[:, 0:7] = command
                env_actions[:, 7] = actions[:, -1]
                
                joint_ids, _ = robot.find_joints(["panda_joint.*"])
                robot.set_joint_position_target(q_des, joint_ids=joint_ids)
                robot.set_joint_velocity_target(command, joint_ids=joint_ids)
                robot.write_data_to_sim()
                
                env.sim.step(render=True)
                env.scene.update(dt=env.physics_dt)
                
                if args_cli.real_robot:
                    time.sleep(0.02)
                last_recovery_state = is_in_recovery

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()