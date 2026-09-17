# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments and APF safety."""

"""Launch Isaac Sim Simulator first."""


#from source.isaaclab_mimic.isaaclab_mimic.envs import franka_stack_ik_rel_blueprint_mimic_env_cfg

from numpy.lib import nanfunctions
from enum import Enum
from PIL import ImageSequence
from six.moves import urllib_robotparser
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
from isaaclab.utils.forwards_kinematics import ForwardsDynamics
#from source.isaaclab.isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.safety.safety_barrier import JP_APF#
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


class Status(Enum):
    OK = 0,
    RISK = 1,
    RECOVERY = 2


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

def get_desired_vel(env, actions, robot, ik_controller):
    ### need to get the command velocity here
    
    current_q = robot.data.joint_pos[:, 0:7]
    hand_body_idx = robot.find_bodies("panda_hand")[0]
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    print(f"[DEBUG] got robot phyxifo pos {ee_pos} quat {ee_quat}")
    # 3. Get Jacobian matrix for the EE link (7 arm joints)
    # Get index of end-effector body frame inside PhysX
    ee_body_idx = ee_frame.body_physx_index if hasattr(ee_frame, "body_physx_index") else 7
    jacobian = robot.root_physx_view.get_jacobians()[:, hand_body_idx, :, 0:7]
    print(f"[DEBUG] Got jacobian and ee body  :{ee_body_idx}")
    # 4. Compute target 7-DOF joint positions via standalone IK
    delta_pose = actions[:, 0:6]  # 6-DOF SE(3) command
    print(f"[DEBUG] delta pose : {delta_pose}")
    ik_controller.set_command(delta_pose, ee_pos=ee_pos, ee_quat=ee_quat)
    print(f"[DEBUG] Managed to set command")
    q_target = ik_controller.compute(
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        jacobian=jacobian,
        joint_pos=current_q,
    )
    print(f"[DEBUG] got q target {q_target}")

    # 5. Compute target 7-DOF joint velocities
    dt = env.physics_dt
    commanded_joint_vel = (q_target - current_q) / dt
    
    # Filter tiny noise
    commanded_joint_vel = torch.where(
        torch.abs(commanded_joint_vel) < 1e-5,
        torch.tensor(0.0, device=commanded_joint_vel.device, dtype=commanded_joint_vel.dtype),
        commanded_joint_vel
    )
    
    # Clamp to Panda joint velocity limits
    QDOT_MAX = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], device=env.device)
    q_dot_des = torch.clamp(commanded_joint_vel, -QDOT_MAX, QDOT_MAX)
    
    return q_dot_des, q_target
   # print(f"[DEBUG] Commanded Joint Velocities: {commanded_joint_vel}")    

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
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=True, 
        ik_method="dls"
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
    print(f"[INFO] Made Controller ")
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
    test_obst = [0.0, 0.0, 0.0, 0.0]
    joint_apf=JP_APF(device=torch.device("cuda:0"),obst=test_obst)
    joint_apf.model.eval()    
    
    field_influence = 0.1
    robot_state = Status.OK

    # Load the trained model
    # if os.path.exists(apf.model_path):
    #     safety_filter.model.load_state_dict(torch.load(apf.model_path))
    # else:
    #     omni.log.error(f"APF model not found at {apf.model_path}")

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
    safety_val =0.1
    is_in_recovery = False
    last_recovery_state = False
    held_q_target = None
    robot = env.unwrapped.scene["robot"]
    # simulate environment
    while simulation_app.is_running():
        try:
            with torch.enable_grad():
                # get device command
                action = teleop_interface.advance()
                actions = action.repeat(env.num_envs, 1).to(env.device)
                teleop_active_input = torch.any(torch.abs(actions[:, 0:6]) > 1e-4)
                # use teleop action if active, otherwise 0 action
                if teleop_active_input:
                    # process actions
                    print(f"[DEBUG] Teleop actions  : {action}")
                    q_dot_des, q_des = get_desired_vel(env, actions, robot, ik_controller)
                    print(f"[DEBUG] calculated desired q_dot  : {q_dot_des}")
                    print(f"[DEBUG] calculated q des : {q_des}")
                    held_q_target = q_des
                else:
                    q_dot_des = torch.zeros((env.num_envs, 7), device=env.device)
                    if held_q_target is None:
                        held_q_target = robot.data.joint_pos[:, 0:7].clone() 
                    q_des = held_q_target
                    print(f"[DEBUG] No teleop command")
                    print(f"[DEBUG] passing q_dot as 0  : {q_dot_des}")
                    print(f"[DEBUG] psssing q des as held_q_target: {q_des}")
                #for i in range(env.num_envs):
                #nominal_velocity = actions[i, 0:3]
                joint_pos = robot.data.joint_pos[0, 0:7]
                print(f"[INFO] joint positon  {joint_pos}")
                joint_vels = robot.data.joint_vel[0,0:7]
                
                safety_val_t0 , joint_vels_t0=joint_apf.get_joint_vals(joint_pos, q_dot_des, 0.009)
                safety_val =safety_val_t0[0].cpu().squeeze().tolist()
                
                if safety_val >=0.001:
                    robot_state = Status.OK
                    wall_color = (0.0, 1.0, 0.0, 1.0)  # Green
                elif safety_val >= 0:
                    robot_state = Status.RISK
                    wall_color = (1.0, 1.0, 0.0, 1.0)  # Yellow - intervention trigger
                else:
                    robot_state = Status.RECOVERY
                    wall_color = (1.0, 0.0, 0.0, 1.0)  # Red
                 #draw box colours
                draw_interface.clear_lines()
                draw_walls(draw_interface, x_min, x_max, y_min, y_max, z_min, z_max, colour=wall_color)
                obs_dict = env.observation_manager.compute_group("policy")
                
                # for ros2 bridge : 
                eef_pos = obs_dict['eef_pos']      # (N, 3)
                eef_quat = obs_dict['eef_quat'] 
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
                            #print(f"DEBUG: Triggering Open Gripper ")
                            bridge.open_gripper()
                    else:
                        gripper_action =-1
                        if gripper_action!=prev_gripper_action:
                            #print(f"DEBUG: Triggering Close Gripper ")
                            bridge.close_gripper()   
                    prev_gripper_action = gripper_action

                #apply actions
                match robot_state:
                    case Status.OK:
                        print(f"[INFO]  Status OK")
                        # if last_recovery_state:
                        #     print(f"[SAFETY] Left safety state")
                        #     buffer_action = torch.tensor([ 0.0, 0.0, 0.0, 0.0, 0.0, -0.0,  0.0], device='cuda:0')
                        #     joint_ids, _ = robot.find_joints(["panda_joint.*"])
                        #     robot.set_joint_velocity_target(buffer_action, joint_ids=joint_ids)
                        #     robot.write_data_to_sim()
                        #     env.sim.step(render=True)
                        #     env.scene.update(dt=env.physics_dt)
                        # else:
                        command = q_dot_des[0]
                        #command = torch.zeros((env.num_envs, 7), device=env.device)
                        # print(f"[INFO] desired motion : {command}")
                        
                        #env.step(actions)
                    case Status.RISK:
                        
                        print(f"[SAFETY] Modifying actions")
                        #print(f"[SAFETY] safety val : {safety_val}")
                        grad = joint_vels_t0[0]                      # dV/dq, points toward safer configs
                        grad_norm = grad.norm()
                        grad_dir = grad / grad_norm if grad_norm > 1e-6 else torch.zeros_like(grad)
                        field_nudge = field_influence * grad_dir
                        # print(f"[SAFETY] desired action : {q_dot_des[0]}")
                        print(f"[SAFETY] Field nudge : {field_nudge}")
                        command = q_dot_des[0] + field_nudge

                        joint_ids, _ = robot.find_joints(["panda_joint.*"])
                        #hold_tensor = torch.full((1, 7), ((1+abs(safety_val))*1000), device=env.device)
                        #recovery_control = torch.mul(joint_vels_t0[0],hold_tensor )
                        #print(f"[SAFETY] modifing velocity term: {command}")
                        #recovery_control = recovery_control + q_dot_des
                        #robot.set_joint_velocity_target(command, joint_ids=joint_ids)
                        safety_val_t0 , joint_vels_t0=joint_apf.get_joint_vals(joint_pos, command, 0.009)
                        print(f"Modified command safety value : {safety_val_t0[0].cpu().squeeze().tolist()}")
                        command = torch.zeros((env.num_envs, 7), device=env.device)
                        #print(f"[SAFETY] sending modified velocity control : {command}")
                        #robot.write_data_to_sim()
                        #env.sim.step(render=True)
                        #env.scene.update(dt=env.physics_dt)

                    case Status.RECOVERY:
                        if not last_recovery_state:
                            print(f"[Triggering safety Overtake]")
                        #print(f"[SAFETY] recovery safety val : {safety_val}")
                        hold_tensor = torch.full((1, 7), (abs(safety_val)*100+0.5), device=env.device)
                        command = torch.mul(joint_vels_t0[0],hold_tensor )
                        command = torch.zeros((env.num_envs, 7), device=env.device)
                        #robot.set_joint_velocity_target(recovery_control, joint_ids=joint_ids)
                        

                ## apply action
                print(f"[INFO] safety val  : {safety_val}")
                env_actions = torch.zeros((env.num_envs, 8), device=env.device)
                print(f"[INFO] placeholder env actions {env_actions}")
                # Assign 7D joint velocity command
                env_actions[:, 0:7] = command
                print(f"[INFO] After adding in velocity commands {env_actions}")
                # Assign 1D gripper action from teleop interface
                env_actions[:, 7] = actions[:, -1]
                print(f"[INFO] Sending control command  {env_actions}")
                # Step environment
                joint_ids, _ = robot.find_joints(["panda_joint.*"])
                robot.set_joint_position_target(q_des, joint_ids=joint_ids)
                robot.set_joint_velocity_target(command, joint_ids=joint_ids)
                # joint_ids, _ = robot.find_joints(["panda_joint.*"])
                # robot.set_joint_velocity_target(command, joint_ids=joint_ids)
                robot.write_data_to_sim()
                applied_torque = robot.data.applied_torque[:, 0:7]
                print(f"[DIAG] applied torque: {applied_torque}")
                env.sim.step(render=True)
                env.scene.update(dt=env.physics_dt)
                
                if args_cli.real_robot:
                    # only sleep if we are really sending on ROS2
                    time.sleep(0.02)
                last_recovery_state = is_in_recovery
                # time.sleep(0.2)
                # else:
                #     env.sim.render()

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
