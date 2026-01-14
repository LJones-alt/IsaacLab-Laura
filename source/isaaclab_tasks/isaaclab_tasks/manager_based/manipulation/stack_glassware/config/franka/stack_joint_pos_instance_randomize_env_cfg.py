# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack_glassware import mdp
from isaaclab_tasks.manager_based.manipulation.stack_glassware.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack_glassware.stack_instance_randomize_env_cfg import (
    StackInstanceRandomizeEnvCfg,
)

from isaaclab_assets.glassware.glassware import ChemistryGlassware

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [0.3281, -0.3684, -0.2787, -2.6138, -2.7527, 2.4999, 0.3331, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cubes_in_focus = EventTerm(
        func=franka_stack_events.randomize_rigid_objects_in_focus,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2")],
            "out_focus_state": torch.tensor([10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "pose_range": {"x": (0.3, 0.65), "y": (-0.35, 0.30), "z": (0.02, 0.02)},
            "min_separation": 0.15,
        },
    )


@configclass
class FrankaCubeStackInstanceRandomizeEnvCfg(StackInstanceRandomizeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Reduce the number of environments due to camera resources
        self.scene.num_envs = 1

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Initialize ChemistryGlassware for creating glassware objects
        glassware = ChemistryGlassware()

        # cube_1: Scale (target placement surface) - two variants at different positions
        cube_1_config_dict = {
            "scale_1": glassware.scale(pos=[0.4, -0.35, 0.01], name="Scale_1"),
            "scale_2": glassware.scale(pos=[10.0, 10.0, 0.01], name="Scale_2"),  # Out of focus position
        }

        # cube_2: Beaker and Conical Flask - positioned far right (positive y)
        cube_2_config_dict = {
            "beaker": glassware.beaker(pos=[0.45, 0.25, 0.02], name="Beaker_1"),
            "flask": glassware.flask(pos=[10.0, 10.0, 0.02], name="Flask_1", scale=(0.7, 0.7, 0.7)),  # Out of focus position
        }

        self.scene.cube_1 = RigidObjectCollectionCfg(rigid_objects=cube_1_config_dict)
        self.scene.cube_2 = RigidObjectCollectionCfg(rigid_objects=cube_2_config_dict)

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
