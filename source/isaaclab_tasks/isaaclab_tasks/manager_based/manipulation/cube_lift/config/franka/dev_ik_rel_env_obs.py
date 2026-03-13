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
from isaaclab_assets.glassware.glassware import ChemistryGlassware
from isaaclab_tasks.manager_based.manipulation.cube_lift import mdp
##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip



@configclass
class FrankaDevEnvCfg(dev_env_cfg.FrankaDevEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # put the beaker on the stir plate
        glassware = ChemistryGlassware()
        self.scene.stirplate = glassware.stirplate(pos=[0.5, 0.0, 0.01])
        self.scene.goal_plate_pos = glassware.scale(pos=[0.4, -0.3, 0.01])
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
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    
                    "panda_joint1":  0.3281,
                    "panda_joint2": -0.3684,   
                    "panda_joint3":  -0.2787,
                    "panda_joint4": -2.6138,  
                    "panda_joint5":  -2.7527,
                    "panda_joint6":  2.4991,  #  +90° → keeps hand level
                    "panda_joint7":  0.3331,
                    "panda_finger_joint1": 0.04,   # open gripper
                    "panda_finger_joint2": 0.04,
                }
            ),
        )
        # replace with relative position controller 
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
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
