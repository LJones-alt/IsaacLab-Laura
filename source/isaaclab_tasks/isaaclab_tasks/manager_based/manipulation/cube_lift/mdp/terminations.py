# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
#from isaaclab.utils.logging_helper import LoggingHelper, ErrorType, LogType
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
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
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    print(f"For DEBUG : DISTANCE TO GOAL : {distance}")
   # if(distance.item() < threshold):
  #      loghelper.logsubtask(LogType.FINISH)
    test = distance < threshold
    if (torch.any(test)):
        print("debug : terminations : ", torch.any(test))
    return distance < threshold


def placed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg=SceneEntityCfg("ee_frame"),
    command_name: str = "object_pose",
    threshold: torch.tensor = torch.tensor([0.05], device="cuda:0"),
    gripper_open_val= torch.tensor([0.038], device="cuda:0"), #open
    atol=0.0001,
    rtol=0.0001,
    gripper_threshold: float = 0.005,
):
    robot: Articulation = env.scene[robot_cfg.name]
    
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name],
    object: RigidObject = env.scene[object_cfg.name]

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    
    object_pos = object.data.root_pos_w
    #end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
   # pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    distance = torch.norm(des_pos_w - object_pos[:, :3], dim=1)
    object_position = distance<threshold
  #  print("object in position ? : ", object_position[0].item())

    
    #print("Gripper state : ",robot.data.joint_pos[:, -1] >  gripper_open_val)
    gripper_open = robot.data.joint_pos[:, -1] >  gripper_open_val
    state = torch.logical_and(object_position, gripper_open)
    return state

def object_stacked_upright(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upper_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),lower_object_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    xy_threshold: float = 0.05, height_threshold: float = 0.1, height_diff: float = 0.05,
    atol=0.0001,
    rtol=0.0001,
    upright_good_deg: float = 30.0, gripper_open_val: torch.Tensor = torch.tensor([0.04]), logging=False) -> torch.Tensor:


    """Stacked AND sufficiently upright (≤ upright_good_deg)."""
    robot: Articulation = env.scene[robot_cfg.name]
    upper: RigidObject = env.scene[upper_object_cfg.name]
    tilt_deg = upright_tilt_deg(upper.data.root_quat_w, upper.data.default_root_state[:, 3:7])
    upright_good = tilt_deg <= upright_good_deg
    stacked = object_stacked(env, robot_cfg, upper_object_cfg, lower_object_cfg,
        xy_threshold, height_threshold, height_diff, gripper_open_val)
    stacked_upright = stacked & upright_good
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        stacked_upright,
    )
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        stacked,
    )

    #if logging:
        #rising = log(env, "object_stacked_upright", stacked_upright, f"Stacked (upright ≤ {upright_good_deg}°)")
       # idx = torch.nonzero(rising, as_tuple=False).squeeze(-1)
        #if idx.numel() > 0:
        #    print(f"[stacked-upright tilt] n={int(idx.numel())} sample={tilt_deg[idx][:5].tolist()}")

    return stacked


def upright_tilt_deg(q_cur, q_init) -> torch.Tensor:
    u_cur, u_init = matrix_from_quat(q_cur)[:, :, 2], matrix_from_quat(q_init)[:, :, 2]
    return torch.rad2deg(torch.acos((u_cur * u_init).sum(-1).clamp(-1.0, 1.0)))


def object_stacked(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upper_object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
    lower_object_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    xy_threshold: float = 0.05, height_threshold: float = 0.1, height_diff: float = 0.05,
    gripper_open_val: torch.Tensor = torch.tensor([0.04]),logging=False) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]
    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    #print(f"For DEBUG : xy_dist : {xy_dist}, height_dist : {height_dist}")
    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)
    # stacked = torch.logical_and(torch.isclose(robot.data.joint_pos[:, -1], 
    #     gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked)
    # stacked = torch.logical_and(torch.isclose(robot.data.joint_pos[:, -2], 
    #     gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked)
    #print(f"For DEBUG : STACKED STATUS : {stacked}")
    return stacked

def object_inserted_upright(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upper_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),lower_object_cfg: SceneEntityCfg = SceneEntityCfg("vialrack"),
    xy_threshold: float = 0.01, height_threshold: float = 0.03, height_diff: float = 0.0,
    atol=0.0001,
    rtol=0.0001,
    upright_good_deg: float = 30.0, gripper_open_val: torch.Tensor = torch.tensor([0.04]), logging=False) -> torch.Tensor:


    """Stacked AND sufficiently upright (≤ upright_good_deg)."""
    robot: Articulation = env.scene[robot_cfg.name]
    upper: RigidObject = env.scene[upper_object_cfg.name]
    tilt_deg = upright_tilt_deg(upper.data.root_quat_w, upper.data.default_root_state[:, 3:7])
    upright_good = tilt_deg <= upright_good_deg
    stacked = object_stacked(env, robot_cfg, upper_object_cfg, lower_object_cfg,
        xy_threshold, height_threshold, height_diff, gripper_open_val)
    #print(f"For DEBUG : STACKED STATUS : {stacked}")
    stacked_upright = stacked & upright_good
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        stacked_upright,
    )
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        stacked_upright,
    )

    #if logging:
        #rising = log(env, "object_stacked_upright", stacked_upright, f"Stacked (upright ≤ {upright_good_deg}°)")
       # idx = torch.nonzero(rising, as_tuple=False).squeeze(-1)
        #if idx.numel() > 0:
        #    print(f"[stacked-upright tilt] n={int(idx.numel())} sample={tilt_deg[idx][:5].tolist()}")

    return stacked