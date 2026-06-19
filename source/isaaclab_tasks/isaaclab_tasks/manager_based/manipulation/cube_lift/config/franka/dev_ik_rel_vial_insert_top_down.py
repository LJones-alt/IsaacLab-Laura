# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab_assets.robots.universal_robots import UR10_CFG
from . import dev_env_cfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.glassware.glassware import ChemistryGlassware
from isaaclab_tasks.manager_based.manipulation.cube_lift import mdp
from isaaclab_tasks.manager_based.manipulation.cube_lift.mdp import franka_stack_events

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG

## add some cameras in



@configclass
class FrankaDevEnvCfg(dev_env_cfg.FrankaDevEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # put the beaker on the stir plate
        glassware = ChemistryGlassware()
        self.scene.stirplate = glassware.stirplate(pos=[0.5, 0.0, 0.01])
        #self.scene.scale = glassware.scale(pos=[0.3, -0.3, 0.01])
        self.scene.vialrack = glassware.line_vial_rack(pos=[0.3, -0.3, 0.0], scale=1.5)
        self.scene.object = glassware.capped_vial(pos=[0.5, 0.0, 0.01], scale =1.0, name="object")
        self.observations.policy.target_object_position = ObsTerm(func=mdp.target_position, params={"object_cfg": SceneEntityCfg("vialrack")})
        
        ### subtask 
        self.observations.subtask_terms.stacked = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("object"),
                "lower_object_cfg": SceneEntityCfg("vialrack"),
            },
        )

        self.events.randomise_object_scale= EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "scale_range": {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_place_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0, 0.2), "y": (0, 0.25), "z": (0.02, 0.02)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
                "asset2_cfg" : SceneEntityCfg("stirplate"),
                "asset3_cfg" : SceneEntityCfg("vialrack")
            },
        )
       # self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
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
                    "panda_finger_joint.*": 0.04,
                },
            ),
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # replace with relative position controller 
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
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
                        rot=[1.0, 1.0, 0.0, 0.0],
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

        self.terminations.success= DoneTerm(func=mdp.object_inserted_upright, params={"lower_object_cfg": SceneEntityCfg("vialrack"), "upright_good_deg": 22.5})
        self.terminations.object_tipped = DoneTerm(func=mdp.object_knocked, params={"max_tilt": 65})
        #self.terminations.success=DoneTerm(func=mdp.object_stacked_upright, params={"lower_object_cfg": SceneEntityCfg("scale")})

        self.observations.subtask_terms.appr_goal=ObsTerm(func=mdp.is_object_lifted, params={"threshold":0.15}
        
    )
       


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
