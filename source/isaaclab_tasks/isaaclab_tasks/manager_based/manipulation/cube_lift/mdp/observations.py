# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab.sensors import FrameTransformer
#from isaaclab.utils.logging_helper import LoggingHelper, ErrorType, LogType

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  #  loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
   
    return object_pos_b

def obstacle_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
  #  loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
   
    return object_pos_b


def reach_object(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
   # loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
   # print("ee pos : ", ee_w)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    #if object_ee_distance.item() < std :
    #    print(f"Observed Object Reached : {object_ee_distance.item()}")
      #  loghelper.logsubtask(LogType.APPR)
    # if object_ee_distance[0] < std:
    #     print(f"Reached object, dist  :{object_ee_distance.item()}")
    return object_ee_distance < std

def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg= SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg= SceneEntityCfg("object"),
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
   # loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )
    #if grasped[0]:
     #   print(f"Observed Object grasped : {grasped.item()}")
    #    loghelper.logsubtask(LogType.GRASP)
    return grasped

def object_released(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.035,
   # loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1]) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] ) > gripper_threshold
    )
   # if grasped[0]:
     #   print(f"Observed Object grasped : {grasped.item()}")
   #     loghelper.logsubtask(LogType.GRASP)
    return grasped

def is_object_lifted(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold : float = 0.05,
    #loghelper : LoggingHelper = LoggingHelper()
):
    #return true when object z coord above a threshold value 
    object = env.scene[obj_cfg.name]
   # if object.data.root_pos_w[:, 2].item() > threshold : 
    #    print(f"Observed Object Lifted : {object.data.root_pos_w[:, 2].item()}")
   #     loghelper.logsubtask(LogType.LIFT)
 
    return object.data.root_pos_w[:, 2] > threshold


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)

def object_near_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  #  loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    des_pos_b = command[:, :3]
   # print(f"desired position : {des_pos_b}, actual position : {object_pos_b}")
    error = torch.norm(des_pos_b - object_pos_b, dim=1)
    #if error<threshold:
    #    loghelper.logsubtask(LogType.FINISH)
    #print("position error : ", error)
    return error < threshold

def object_goal_norm_error(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
   # loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    des_pos_b = command[:, :3]
   # print(f"desired position : {des_pos_b}, actual position : {object_pos_b}")
    error = torch.norm(des_pos_b - object_pos_b, dim=1)
    #print("norm position error : ", error)
    return error 

def robot_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot: Articulation = env.scene[robot_cfg.name]
   # print(robot.data.joint_pos)
    return torch.tensor([0.04])

def position_command_error(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "object_pose",
  #  loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    des_pos_b = command[:, :3]
   # print(f"desired position : {des_pos_b}, actual position : {object_pos_b}")
    error = torch.sub(des_pos_b, object_pos_b)
    #print("position error : ", error)
    return error

def get_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    #print("Joint positions: : ", asset.data.joint_pos[:, asset_cfg.joint_ids])
    return asset.data.joint_pos[:, asset_cfg.joint_ids]

# def logstep(
#         env: ManagerBasedRLEnv,
#   #      loghelper : LoggingHelper
#     ):
#     loghelper.logstep()
#     return torch.tensor([0])
def upright_tilt_deg(q_cur, q_init) -> torch.Tensor:
    u_cur, u_init = matrix_from_quat(q_cur)[:, :, 2], matrix_from_quat(q_init)[:, :, 2]
    return torch.rad2deg(torch.acos((u_cur * u_init).sum(-1).clamp(-1.0, 1.0)))
 
 
 
def object_tilt (env: ManagerBasedRLEnv, upper_object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    object: RigidObject = env.scene[upper_object_cfg.name]
    tilt_deg = upright_tilt_deg(object.data.root_quat_w, object.data.default_root_state[:, 3:7])
    #print("obs term : tilt : ", tilt_deg)
    return tilt_deg

def object_knocked(env: ManagerBasedRLEnv, upper_object_cfg: SceneEntityCfg = SceneEntityCfg("object"), max_tilt: float = 90):
    object: RigidObject = env.scene[upper_object_cfg.name]
    tilt_deg = upright_tilt_deg(object.data.root_quat_w, object.data.default_root_state[:, 3:7])
    if tilt_deg.cpu().detach().numpy()[0] > max_tilt:
        print("object knocked")
        return torch.tensor([True], device='cuda:0')
    else :return torch.tensor([False], device='cuda:0')