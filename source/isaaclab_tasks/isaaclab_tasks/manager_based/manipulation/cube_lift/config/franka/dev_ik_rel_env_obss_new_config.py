# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from . import dev_env_cfg
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 
from isaaclab_assets.glassware.glassware import ChemistryGlassware
from isaaclab_tasks.manager_based.manipulation.cube_lift import mdp
##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, PANDA_OFFSET_ROBOTIQ_CFG # isort: skip



@configclass
class FrankaDevEnvCfg(dev_env_cfg.FrankaDevEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # put the beaker on the stir plate
        glassware = ChemistryGlassware()
        self.scene.stirplate = glassware.stirplate(pos=[0.5, 0.0, 0.01])
        self.scene.goal_plate_pos = glassware.scale(pos=[0.4, -0.3, 0.01])
        
        # quickly change the observations
        self.observations.policy.eef_pos = ObsTerm(func=mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        self.observations.policy.eef_quat = ObsTerm(func=mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        #self.observations.policy.gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        #self.observations.policy.gripper_width = ObsTerm(func=mdp.gripper_pos, params={"ee_frame_cfg": "robotiq_arg2f_base_link"})
        self.commands.object_pose.body_name = "robotiq_arg2f_base_link"
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         'finger_joint', 
        #         'right_outer_knuckle_joint', 
        #         'left_inner_finger_joint', 
        #         'right_inner_finger_joint'
        #     ],
        #     open_command_expr={
        #         'finger_joint': 0.0,
        #         'right_outer_knuckle_joint': 0.0,
        #         'left_inner_finger_joint': 0.0,
        #         'right_inner_finger_joint': 0.0
        #     },
        #     close_command_expr={
        #         'finger_joint': 0.8,
        #         'right_outer_knuckle_joint': 0.8,
        #         'left_inner_finger_joint': 0.8,
        #         'right_inner_finger_joint': 0.8
        #     },
        # )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                'finger_joint', 
                'right_outer_knuckle_joint', 
                'left_inner_finger_joint', 
                'right_inner_finger_joint'
            ],
            open_command_expr={
                'finger_joint': 0.0,
                'right_outer_knuckle_joint': 0.0,
                'left_inner_finger_joint': 0.0,
                'right_inner_finger_joint': 0.0
            },
            close_command_expr={
                'finger_joint': 0.8,
                'right_outer_knuckle_joint': 0.8,
                # FEEDBACK FIX: Change these to -0.8 to maintain parallelism
                'left_inner_finger_joint': -0.8, 
                'right_inner_finger_joint': -0.8
            },
        )

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_multiple_root_state_uniform,
            mode="reset",
            #[0.54, -0.3, 0.0]
            params={
                "pose_range": {"x": (0, 0.2), "y": (0, 0.25), "z": (0.02, 0.02)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "asset2_cfg" : SceneEntityCfg("stirplate")
            },
        )
        self.scene.robot = PANDA_OFFSET_ROBOTIQ_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.037,
                    "panda_joint7": 0.741,
                }
            ),
        )
        # replace with relative position controller 
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="robotiq_arg2f_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),# rot=[0.7071, -0.7071, 0.0, 0.0]),
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg= marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_inner_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        self.observations.subtask_terms.appr_goal = ObsTerm(
            func=mdp.object_near_goal,
            params={ 
                "threshold": 0.05, 
                "command_name": "object_pose",
            },
        )
        #self.observations.policy.target_object_position = ObsTerm(func=mdp.generated_command_position, params={"command_name": "object_pose"})
        self.terminations.success = DoneTerm(func=mdp.object_near_goal, params={"threshold": 0.05 })  

       


@configclass
class FrankaCubeEnvCfg_PLAY(FrankaDevEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
