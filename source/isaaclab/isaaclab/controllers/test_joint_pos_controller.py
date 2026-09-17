# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


class TestJointPosController:
    """Joint Position Controller for robot articulation joints.

    This controller receives joint position commands (either absolute joint positions or relative delta joint positions)
    and computes the target joint position commands to send to the robot articulation drivers.
    """

    def __init__(self, cfg: TestJointPosControllerCfg, num_envs: int, num_joints: int = 7, device= "cuda:0"):
        """Initialize the joint position controller.

        Args:
            cfg: Configuration for the controller.
            num_envs: Number of environment instances.
            num_joints: Number of actuated robot joints/DOFs.
            device: Computing device ('cuda' or 'cpu').
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_joints = num_joints
        self._device = device

        # Buffers
        self._command = torch.zeros(self.num_envs, self.num_joints, device=self._device)
        self.joint_pos_des = torch.zeros(self.num_envs, self.num_joints, device=self._device)

        # Scale tensor
        if isinstance(self.cfg.scale, (int, float)):
            self._scale = torch.full((1, self.num_joints), float(self.cfg.scale), device=self._device)
        else:
            self._scale = torch.tensor(self.cfg.scale, device=self._device).unsqueeze(0)

        # Offsets buffer
        self._dof_pos_offset = torch.zeros(self.num_envs, self.num_joints, device=self._device)
        if self.cfg.dof_pos_offset is not None:
            self._dof_pos_offset[:] = torch.tensor(self.cfg.dof_pos_offset, device=self._device)

    @property
    def action_dim(self) -> int:
        """Dimension of the input command tensor."""
        return self.num_joints

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset the internal command buffers.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            self._command.zero_()
            self.joint_pos_des.zero_()
        else:
            self._command[env_ids] = 0.0
            self.joint_pos_des[env_ids] = 0.0

    def set_command(self, command: torch.Tensor, current_joint_pos: torch.Tensor | None = None):
        """Set the target joint position command.

        Args:
            command: Input command tensor of shape (num_envs, num_joints).
            current_joint_pos: Current joint position tensor of shape (num_envs, num_joints).
                Required if `use_relative_mode` is True.

        Raises:
            ValueError: If `command` shape does not match expected shape.
            ValueError: If relative mode is used but `current_joint_pos` is None.
        """
        if command.ndim == 1:
            command = command.unsqueeze(0)

        if command.shape != (self.num_envs, self.num_joints):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_envs, self.num_joints)}'."
            )

        self._command[:] = command * self._scale

        if self.cfg.use_relative_mode:
            if current_joint_pos is None:
                raise ValueError("`current_joint_pos` cannot be None when `use_relative_mode` is True!")
            self.joint_pos_des[:] = current_joint_pos + self._command + self._dof_pos_offset
        else:
            self.joint_pos_des[:] = self._command + self._dof_pos_offset

    def compute(
        self,
        current_joint_pos: torch.Tensor | None = None,
        dof_pos_limits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the target joint positions to send to the articulation.

        Args:
            current_joint_pos: Optional current joint position tensor.
            dof_pos_limits: Optional joint limits tensor of shape (num_envs, num_joints, 2) or (num_joints, 2)
                used to clamp desired joint positions.

        Returns:
            Target joint positions tensor of shape (num_envs, num_joints).
        """
        target = self.joint_pos_des.clone()

        if dof_pos_limits is not None:
            if dof_pos_limits.ndim == 2:
                lower = dof_pos_limits[:, 0]
                upper = dof_pos_limits[:, 1]
            elif dof_pos_limits.ndim == 3:
                lower = dof_pos_limits[..., 0]
                upper = dof_pos_limits[..., 1]
            else:
                raise ValueError(f"Invalid dof_pos_limits shape: {dof_pos_limits.shape}")

            target = torch.clamp(target, min=lower, max=upper)

        return target


@configclass
class TestJointPosControllerCfg:
    """Configuration for test joint position controller."""

    class_type: type = TestJointPosController
    """The associated controller class."""

    command_type: Literal["absolute", "relative"] = "absolute"
    """Type of joint position command.
    
    - "absolute": Input actions directly specify target joint positions.
    - "relative": Input actions specify delta changes to current joint positions.
    """

    use_relative_mode: bool = False
    """Whether to use relative mode for the controller. Defaults to False."""

    scale: float | Sequence[float] = 1.0
    """Scaling factor applied to the input command. Defaults to 1.0."""

    dof_pos_offset: Sequence[float] | None = None
    """Offset added to joint position targets. Defaults to None."""

    def __post_init__(self):
        if self.command_type not in ["absolute", "relative"]:
            raise ValueError(f"Unsupported command_type '{self.command_type}'. Supported modes: ['absolute', 'relative'].")
        if self.command_type == "relative":
            self.use_relative_mode = True
