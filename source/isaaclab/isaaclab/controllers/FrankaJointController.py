import torch
from dataclasses import MISSING
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.config import configclass


class FrankaJointControllerAction(ActionTerm):
    cfg: "FrankaJointControllerActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "FrankaJointControllerActionCfg", env):
        super().__init__(cfg, env)

        # Resolve robot articulation and target joint indices
        self._asset: Articulation = env.scene[self.cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        # Pre-allocate GPU tensor buffers
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Initialize PD gain tensors for velocity/torque calculations
        if isinstance(self.cfg.p_gains, (float, int)):
            self.kp = torch.full((self._num_joints,), float(self.cfg.p_gains), device=self.device)
        else:
            self.kp = torch.tensor(self.cfg.p_gains, device=self.device)

        if isinstance(self.cfg.d_gains, (float, int)):
            self.kd = torch.full((self._num_joints,), float(self.cfg.d_gains), device=self.device)
        else:
            self.kd = torch.tensor(self.cfg.d_gains, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._num_joints  # 7 arm joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Processes incoming 7-DOF joint targets from the state machine."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self.cfg.scale + self.cfg.offset

    def apply_actions(self):
        """Executes low-level control commands on PhysX during decimation steps."""
        if self.cfg.control_mode == "position":
            # Direct PhysX implicit PD joint drive targets
            self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

        elif self.cfg.control_mode == "velocity":
            # Proportional velocity targets: v_des = Kp * (q_des - q)
            q_curr = self._asset.data.joint_pos[:, self._joint_ids]
            v_des = self.kp * (self._processed_actions - q_curr)
            self._asset.set_joint_velocity_target(v_des, joint_ids=self._joint_ids)

        elif self.cfg.control_mode == "torque":
            # Low-level joint space PD effort: tau = Kp*(q_des - q) - Kd*q_dot
            q_curr = self._asset.data.joint_pos[:, self._joint_ids]
            qd_curr = self._asset.data.joint_vel[:, self._joint_ids]

            torques = self.kp * (self._processed_actions - q_curr) - self.kd * qd_curr

            if self.cfg.clip_torques is not None:
                torques = torch.clamp(torques, -self.cfg.clip_torques, self.cfg.clip_torques)

            self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)
        else:
            raise ValueError(f"Invalid control mode specified: {self.cfg.control_mode}")

@configclass
class FrankaJointControllerActionCfg(ActionTermCfg):
    class_type: type = FrankaJointControllerAction
    asset_name: str = "robot"
    joint_names: list[str] = ["panda_joint[1-7]"]
    control_mode: str = "position"  # Options: "position", "velocity", "torque"
    p_gains: list[float] | float = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
    d_gains: list[float] | float = [50.0, 50.0, 50.0, 50.0, 30.0, 20.0, 10.0]
    clip_torques: float | None = 87.0
    scale: float = 1.0
    offset: float = 0.0


class FrankaGripperBinaryAction(ActionTerm):
    """Maps 1-DOF binary command (0.0=OPEN, 1.0=CLOSE) to 2 finger position targets."""

    cfg: "FrankaGripperBinaryActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "FrankaGripperBinaryActionCfg", env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[self.cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1  # 1-DOF binary trigger from FSM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        is_close = self._raw_actions > 0.5
        target_pos = torch.where(is_close, self.cfg.close_pos, self.cfg.open_pos)
        self._processed_actions[:] = target_pos.repeat(1, self._num_joints)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class FrankaGripperBinaryActionCfg(ActionTermCfg):
    """Configuration for Franka 2-finger binary gripper."""
    class_type: type = FrankaGripperBinaryAction
    asset_name: str = "robot"
    joint_names: list[str] = ["panda_finger_joint1", "panda_finger_joint2"]
    open_pos: float = 0.04
    close_pos: float = 0.0