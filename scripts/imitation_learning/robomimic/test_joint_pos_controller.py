# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint Position State Machine Controller for Pick-and-Place tasks."""

from enum import Enum
import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.controllers import TestJointPosController, TestJointPosControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms


class TaskState(Enum):
    REST = 0
    APPROACH_ABOVE_OBJECT = 1
    APPROACH_OBJECT = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4
    MIDPOINT = 5
    APPROACH_ABOVE_GOAL = 6
    APPROACH_GOAL = 7
    UNGRASP_OBJECT = 8


class GripperState:
    """States for the gripper."""
    OPEN = 0.0
    CLOSE = 1.0


class TestJointPosControllerSM:
    """State Machine Controller producing joint-position actions for robot manipulation tasks."""

    def __init__(self, device, env, use_relative_joint_mode: bool = False):
        self.device = device
        self.env = env
        self.use_relative_joint_mode = use_relative_joint_mode

        self.object_start_pos = torch.tensor([0, 0, 0], device=self.device)
        self.object_goal_pos = torch.tensor([0, 0, 0], device=self.device)
        self.object_start_rot = torch.tensor([0, 0, 0, 0], device=self.device)
        self.object_goal_rot = torch.tensor([0, 0, 0, 0], device=self.device)
        self.current_ee_pos = torch.tensor([0, 0, 0], device=self.device)
        self.current_ee_rot = torch.tensor([0, 0, 0, 0], device=self.device)

        self.rest_pos = torch.tensor([[0.5206, 0.0096, 0.3751]], device=self.device)
        self.rest_rot = torch.tensor([[0.4821, 0.4953, -0.5114, -0.5106]], device=self.device)
        self.ee_offset = torch.tensor([[0, 0, 0.012]], device=self.device)

        self.robot = env.unwrapped.scene["robot"]
        self.root_pose_w = self.robot.data.root_pose_w

        self.state = TaskState.REST
        self.gripper_state = GripperState.OPEN
        self.desired_action = torch.zeros((1, 8), device=self.device)

        self.threshold = 0.05
        self.clamp = 0.05
        self.state_timer = 50
        self.state_timer_reset = 50

        self.offset_pos = torch.tensor([[0.14, 0.0, 0.0]], device=self.device)
        self.offset_quat = torch.tensor([[0.5, -0.5, 0.5, -0.5]], device=self.device)

        self.robot_entity_cfg, self.ee_jacobi_idx = self._setup_robot()
        self.num_arm_joints = len(self.robot_entity_cfg.joint_ids)

        # Differential IK Controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        )
        self.ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=self.device)

        # Joint Position Processing Controller
        jpos_cfg = TestJointPosControllerCfg(
            command_type="relative" if self.use_relative_joint_mode else "absolute",
            scale=1.0,
        )
        self.joint_pos_controller = TestJointPosController(
            cfg=jpos_cfg,
            num_envs=1,
            num_joints=self.num_arm_joints,
            device=self.device,
        )

        self._init_task()

    def _update_joint_pos(self):
        """Updated to use Isaac Lab ArticulationData properties."""
        current_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        current_joint_vel = self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids]
        current_joint_acc = self.robot.data.joint_acc[:, self.robot_entity_cfg.joint_ids]
        current_joint_torque = self.robot.data.applied_torque[:, self.robot_entity_cfg.joint_ids]

    def _setup_robot(self):
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg.resolve(self.env.unwrapped.scene)
        if self.robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        return robot_entity_cfg, ee_jacobi_idx

    def _init_task(self):
        obj_pos_w = self.env.unwrapped.scene["object"].data.root_pose_w[:, :3]
        obj_quat_w = self.env.unwrapped.scene["object"].data.root_pose_w[:, 3:7]
        obj_pos_b, _ = subtract_frame_transforms(
            self.root_pose_w[:, :3], self.root_pose_w[:, 3:7],
            obj_pos_w, obj_quat_w
        )
        self.object_start_pos = obj_pos_b[:, :3]

        goal_pos_w = self.env.unwrapped.scene["scale"].data.root_pose_w[:, :3]
        goal_quat_w = self.env.unwrapped.scene["scale"].data.root_pose_w[:, 3:7]
        goal_pos_b, _ = subtract_frame_transforms(
            self.root_pose_w[:, :3], self.root_pose_w[:, 3:7],
            goal_pos_w, goal_quat_w
        )
        self.object_goal_pos = goal_pos_b[:, :3]

        self.object_start_rot = self.rest_rot
        self.object_goal_rot = self.rest_rot
        self._update_ee_pose()

    def _update_ee_pose(self):
        raw_link_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        link_pos_w = raw_link_pose_w[:, 0:3]
        link_quat_w = raw_link_pose_w[:, 3:7]
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            link_pos_w, link_quat_w,
            self.offset_pos, self.offset_quat
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            self.root_pose_w[:, 0:3], self.root_pose_w[:, 3:7],
            ee_pos_w, ee_quat_w
        )
        self.current_ee_pos = ee_pos_b
        self.current_ee_rot = ee_quat_b

    def _get_goal_pose(self):
        match self.state:
            case TaskState.REST:
                self.gripper_state = GripperState.OPEN
                self._update_ee_pose()
                self._update_joint_pos()
                if self.current_ee_pos[0, 2].item() < 0.2:
                    pos = torch.add(self.current_ee_pos, torch.tensor([[0, 0, 0.2]], device=self.device))
                    return pos, self.rest_rot
                return self.rest_pos, self.rest_rot

            case TaskState.APPROACH_ABOVE_OBJECT:
                self.gripper_state = GripperState.OPEN
                position = torch.add(self.object_start_pos, torch.tensor([[0, 0, 0.1]], device=self.device))
                return position, self.rest_rot

            case TaskState.APPROACH_OBJECT:
                self.gripper_state = GripperState.OPEN
                position = torch.add(self.object_start_pos, self.ee_offset)
                return position, self.rest_rot

            case TaskState.GRASP_OBJECT:
                self.gripper_state = GripperState.CLOSE
                position = torch.add(self.object_start_pos, self.ee_offset)
                return position, self.rest_rot

            case TaskState.LIFT_OBJECT:
                self.gripper_state = GripperState.CLOSE
                position = torch.add(self.object_start_pos, torch.tensor([[0, 0, 0.1]], device=self.device))
                return position, self.rest_rot

            case TaskState.MIDPOINT:
                self.gripper_state = GripperState.CLOSE
                position = torch.add(self.object_start_pos, torch.tensor([[0, 0, 0.2]], device=self.device))
                return position, self.rest_rot

            case TaskState.APPROACH_ABOVE_GOAL:
                self.gripper_state = GripperState.CLOSE
                position = torch.add(self.object_goal_pos, torch.tensor([[0, 0, 0.3]], device=self.device))
                return position, self.rest_rot

            case TaskState.APPROACH_GOAL:
                self.gripper_state = GripperState.CLOSE
                position = torch.add(self.object_goal_pos, torch.tensor([[0, 0, 0.1]], device=self.device))
                return position, self.rest_rot

            case TaskState.UNGRASP_OBJECT:
                self.gripper_state = GripperState.OPEN
                position = torch.add(self.object_goal_pos, torch.tensor([[0, 0, 0.08]], device=self.device))
                return position, self.rest_rot

    def _calc_joint_action(self, desired_pos: torch.Tensor, desired_rot: torch.Tensor) -> torch.Tensor:
        if desired_pos.dim() == 1:
            desired_pos = desired_pos.unsqueeze(0)
        if desired_rot.dim() == 1:
            desired_rot = desired_rot.unsqueeze(0)

        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        jacobians = self.robot.root_physx_view.get_jacobians()
        jacobian = jacobians[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]

        pose_cmd = torch.cat([desired_pos, desired_rot], dim=-1)
        self.ik_controller.set_command(pose_cmd)

        target_joint_pos = self.ik_controller.compute(
            ee_pos=self.current_ee_pos,
            ee_quat=self.current_ee_rot,
            jacobian=jacobian,
            joint_pos=joint_pos,
        )
        self._last_target_joint_pos = target_joint_pos

        if self.use_relative_joint_mode:
            delta_q = target_joint_pos - joint_pos
            self.joint_pos_controller.set_command(delta_q, current_joint_pos=joint_pos)
            arm_action = self.joint_pos_controller.compute(
                current_joint_pos=joint_pos,
                dof_pos_limits=self.robot.data.soft_joint_pos_limits[:, self.robot_entity_cfg.joint_ids],
            )
        else:
            self.joint_pos_controller.set_command(target_joint_pos)
            arm_action = self.joint_pos_controller.compute(
                dof_pos_limits=self.robot.data.soft_joint_pos_limits[:, self.robot_entity_cfg.joint_ids],
            )

        gripper_action = torch.tensor([[self.gripper_state]], device=self.device)
        self.desired_action = torch.cat([arm_action, gripper_action], dim=-1)
        return self.desired_action

    def get_action(self) -> torch.Tensor:
        self._update_ee_pose()
        desired_pos, desired_rot = self._get_goal_pose()
        action = self._calc_joint_action(desired_pos, desired_rot)

        pos_dist = torch.norm(self.current_ee_pos - desired_pos)
        rot_dist = torch.norm(self.current_ee_rot - desired_rot)

        match self.state:
            case TaskState.REST:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < self.state_timer_reset:
                        self.state = TaskState.APPROACH_ABOVE_OBJECT
                        self.state_timer = self.state_timer_reset

            case TaskState.APPROACH_ABOVE_OBJECT:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < self.state_timer_reset:
                        self.state = TaskState.APPROACH_OBJECT
                        self.state_timer = self.state_timer_reset

            case TaskState.APPROACH_OBJECT:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < self.state_timer_reset / 2:
                        self.state = TaskState.GRASP_OBJECT
                        self.state_timer = self.state_timer_reset

            case TaskState.GRASP_OBJECT:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < (self.state_timer_reset * 0.9):
                        self.state = TaskState.LIFT_OBJECT
                        self.state_timer = self.state_timer_reset

            case TaskState.LIFT_OBJECT:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < self.state_timer_reset:
                        self.state = TaskState.MIDPOINT
                        self.state_timer = self.state_timer_reset

            case TaskState.MIDPOINT:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < self.state_timer_reset:
                        self.state = TaskState.APPROACH_ABOVE_GOAL
                        self.state_timer = self.state_timer_reset
                        self._init_task()

            case TaskState.APPROACH_ABOVE_GOAL:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < self.state_timer_reset:
                        self.state = TaskState.APPROACH_GOAL
                        self.state_timer = self.state_timer_reset

            case TaskState.APPROACH_GOAL:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < self.state_timer_reset / 2:
                        self.state = TaskState.UNGRASP_OBJECT
                        self.state_timer = self.state_timer_reset

            case TaskState.UNGRASP_OBJECT:
                if pos_dist < self.threshold and rot_dist < self.threshold:
                    self.state_timer -= 1
                    if self.state_timer < 1:
                        self.state = TaskState.REST
                        self.state_timer = self.state_timer_reset

            case _:
                self.state = TaskState.REST

        return action

    def reset(self):
        self.state = TaskState.REST
        self.state_timer = self.state_timer_reset
        self._init_task()


TestController = TestJointPosControllerSM