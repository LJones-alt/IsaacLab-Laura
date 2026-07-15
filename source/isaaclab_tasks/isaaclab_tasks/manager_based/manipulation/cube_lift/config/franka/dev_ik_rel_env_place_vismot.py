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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.glassware.glassware import ChemistryGlassware
from isaaclab.markers.config import FRAME_MARKER_CFG 
from isaaclab_tasks.manager_based.manipulation.cube_lift import mdp
from isaaclab_tasks.manager_based.manipulation.cube_lift.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.cube_lift.lift_env_cfg import CubeEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import numpy as np
import torch
from isaaclab.envs.mdp.actions.binary_joint_actions import *# BinaryJointPositionAction, BinaryJointPositionActionCfg
##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG , PANDA_OFFSET_ROBOTIQ_CFG, FRANKA_ROBOTIQ_GRIPPER_CFG, TEST# isort: skip
#from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG

## add some cameras in

## this is fully configured for cosmos now! 

@configclass
class EventCfg():
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomise_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        # params={
        #     "scale_range": {"x": (0.4, 0.7), "y": (0.4, 0.7), "z": (0.5, 0.7)},
        #     "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        # },
        params={
            "scale_range": {"x": (0.5, 0.5), "y": (0.5, 0.5), "z": (0.6, 0.6)},
            "asset_cfg": SceneEntityCfg("object", body_names="object"),
        },
    #     params={
    #         "scale_range": {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},
    #         "asset_cfg": SceneEntityCfg("object", body_names="object"),
    #     },
    )
    # randomize_light = EventTerm(
    #     func=franka_stack_events.randomize_scene_lighting_domelight,
    #     mode="reset",
    #     params={
    #         "intensity_range": (1500.0, 10000.0),
    #         "color_variation": 0.4,
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
    #         ],
    #         "default_intensity": 3000.0,
    #         "default_color": (0.75, 0.75, 0.75),
    #         "default_texture": "",
    #     },
    # )

    # randomize_table_visual_material = EventTerm(
    #     func=franka_stack_events.randomize_visual_texture_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("table"),
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    #         ],
    #         "default_texture": (
    #             f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png"
    #         ),
    #     },
    # )
    reset_object_position = EventTerm(
        func=mdp.reset_place_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0,0), "y": (0,0), "z": (0.02, 0.02)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="object"),
            "asset2_cfg" : SceneEntityCfg("stirplate"),
            "asset3_cfg" : SceneEntityCfg("scale")
        },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # for debugging 
       # abs_joint_pos = ObsTerm(func=mdp.get_joint_pos)

        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        #cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        #cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        target_object_position = ObsTerm(func=mdp.target_position, params={"object_cfg": SceneEntityCfg("scale")})
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )


        # for cosmos add the segmentation, normals and depth data
        table_cam_segmentation= ObsTerm(
            func = mdp.image,
            params = {
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "semantic_segmentation",
                "normalize" : True,
            }
        )

        table_cam_normals= ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type" : "normals",
                "normalize" : True,
            }
        )

        table_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "distance_to_image_plane",
                "normalize" : True,
            }
        )

        ### debug obs


        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""
        grasp = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        lift = ObsTerm(
            func=mdp.is_object_lifted,
            params={
                "obj_cfg": SceneEntityCfg("object"),
                "threshold" : 0.1
            }
        )
        stacked = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("object"),
                "lower_object_cfg": SceneEntityCfg("scale"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class FrankaDevEnvVMCfg(dev_env_cfg.FrankaDevEnvCfg):
   
    def __post_init__(self):
        # post init of parent
        self.observations = ObservationsCfg()
        self.events: EventCfg = EventCfg()
        # Evaluation settings - maybe fix these ? 
        self.eval_mode = False
        self.eval_type = None
        print("\n" + "="*50 + "\n[DEBUG] LOADED dev_ik_rel_env_place_vismot\n" + "="*50 + "\n")
        from isaaclab.sensors import FrameTransformerCfg
        from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

        self.scene.ee_frame = None
        super().__post_init__()
       

        # Update the ee_frame to use the correct prim paths for the custom robot
                # self.scene.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_rf85_gripper2/panda_link0",
        #     debug_vis=False,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_rf85_gripper2/panda_link7",
        #             name="end_effector",
        #             offset=OffsetCfg(
        #                 pos=[0.0, 0.0, 0.22], # Increased offset to reach gripper tip from link7
        #             ),
        #         ),
        #     ],
        # )

        import carb
        from isaacsim.core.utils.carb import set_carb_setting

        carb_setting = carb.settings.get_settings()
        #set_carb_setting(carb_setting, "/rtx/domeLight/upperLowerStrategy", 4)
        # Force Path Tracing via Carb
    #     set_carb_setting(carb_setting, "/rtx/renderMode", "PathTracing")
    #     set_carb_setting(carb_setting, "/rtx/pathtracing/spp", 1)
    #     set_carb_setting(carb_setting, "/rtx/pathtracing/totalSpp", 1)
        
    #     self.sim.render_settings = {
    #         "rtx/renderMode": "PathTracing",
    #         "rtx/pathtracing/spp": 1,
    #         "rtx/pathtracing/totalSpp": 1,
    #         "rtx/pathtracing/maxBounces": 4,
    #         "rtx/hydra/enabled": True,
    #     }
    #   # apply_semantic_label(self.scene.robot.prim_path, "Robot")
       # apply_semantic_label(self.scene.table.prim_path, "table")
        SEMANTIC_MAPPING = {
            "class:object": (120, 230, 255, 255),
            "class:stirplate": (255, 36, 66, 255),
            "class:scale": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (204, 110, 248, 255),
            "class:Robot": (204, 110, 248, 255),
            "class:UNLABELLED": (150, 150, 150, 255),
            "class:BACKGROUND": (200, 200, 200, 255),
        }

        ## add some extra tables
        self.scene.table2 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table2",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0.75, 0], rot=[1.0, 0, 0, 0]),
            #init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
            #spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )
        self.scene.Table3 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table3",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, -0.75, 0], rot=[1.0, 0, 0, 0]),
            #init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
            #spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )
        
        # put the beaker on the stir plate
        glassware = ChemistryGlassware()
        self.scene.stirplate = glassware.IKAplate(pos=[0.1, 0.0, 0.01])
        self.scene.scale = glassware.scale(pos=[0.1, -0.4, 0.01])
        self.scene.object = glassware.beaker(pos=[0.2, 0.0, 0.01], name="object")
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=96,
            width=96,
            data_types=["rgb", "semantic_segmentation", "normals", "distance_to_image_plane"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            ### Where is this in the scene?
            offset=CameraCfg.OffsetCfg(
                pos=(1.6, 0.0, 1.0), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )
        self.scene.wrist_cam = CameraCfg(
            #prim_path="{ENV_REGEX_NS}/Robot/wrist_cam",
            prim_path="{ENV_REGEX_NS}/Robot/franka_robotiq_2f_85_flattened/Gripper/Robotiq_2F_85/base_link/wrist_camera_flipped",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb", "distance_to_image_plane"],
            # spawn=sim_utils.PinholeCameraCfg(
            #     focal_length=2.4, focus_distance=28.0, horizontal_aperture=5.3, clipping_range=(0.1, 2)
            # ),
            spawn=None,
            # offset=CameraCfg.OffsetCfg(
            #     pos=(0.011, -0.031, 0.4), rot=(-0.4, 0.58, 0.57, -0.418), convention="ros"
            # ),
        )
        self.viewer.eye = (-1.5, -0.3, 1.0)
        self.rerender_on_reset = True
        # self.sim.render_settings = {
        #     "rtx/renderMode": "PathTracing",
        #     "rtx/pathtracing/spp": 1,
        #     "rtx/pathtracing/totalSpp": 1,
        #     "rtx/pathtracing/maxBounces": 4,
        #     "rtx/hydra/enabled": True,
        # }
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss
        # self.sim.dt=0.0002
        # self.sim.render_interval=4
        #self.sim.render.antialiasing_mode = "DLSS"  # enable dlss for cleaner PathTracing

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]


        #self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Create a deep copy to boost the default 0.0 stiffness of the passive gripper joints
        # import copy
        # robot_custom_cfg = copy.deepcopy(PANDA_OFFSET_ROBOTIQ_CFG)
        # if "gripper_passive" in robot_custom_cfg.actuators:
        #     robot_custom_cfg.actuators["gripper_passive"].stiffness = 11.25
        #     robot_custom_cfg.actuators["gripper_passive"].effort_limit_sim = 10.0
        # if "gripper_finger" in robot_custom_cfg.actuators:
        #     robot_custom_cfg.actuators["gripper_finger"].stiffness = 11.25
        #     robot_custom_cfg.actuators["gripper_finger"].effort_limit_sim = 10.0

        # self.scene.robot = robot_custom_cfg.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        #     init_state=ArticulationCfg.InitialStateCfg(
        #         joint_pos={
        #             "panda_joint1": 0.0,
        #             "panda_joint2": -0.569,
        #             "panda_joint3": 0.0,
        #             "panda_joint4": -2.810,
        #             "panda_joint5": 0.0,
        #             "panda_joint6": 3.037,
        #             "panda_joint7": 0.741,
        #             "finger_joint": 0.0,
        #             # ".*_inner_finger_joint": 0.0,
        #             # ".*_inner_finger_pad_joint": 0.0,
        #             # ".*_outer_.*_joint": 0.0,

        #         },
        #     ),
        # )

        self.scene.robot=TEST.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        # replace with relative position controller 
        # self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         "panda_joint1", "panda_joint2", "panda_joint3", 
        #         "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        #     ],
        #     body_name="panda_link7",
        #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        #     scale=0.5,
        #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.22]),# rot=[0.7071, -0.7071, 0.0, 0.0]),
        # )

        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     # ONLY target the main driven joint here
        #     joint_names=["finger_joint"],
        #     open_command_expr={
        #         "finger_joint": 0.0,
        #     },
        #     close_command_expr={
        #         # Note: verify if Robotiq 140 takes radians or linear meters in your USD. 
        #         # If it's radians, 46.0 is way too high. Usually it's ~0.7 to 0.8 radians.
        #         "finger_joint": 0.7, 
        #     },
        # )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="base_link",  # Robotiq 2F-85 base flange (gripper mount); matches ee_pos/ee_quat helpers
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            # Robotiq 2F-85 max height base flange -> fingertip is 162.8mm (per Robotiq spec).
            # Uncomment to control the fingertip plane instead of the base flange.
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1628]),
        )

        self.actions.gripper_action = BinaryJointPositionZeroToOneActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": np.pi / 4},
        )
        
        # Match this here as well
        self.gripper_joint_names = ["finger_joint"]
        self.gripper_open_val = 0.0
        self.gripper_threshold = 0.005
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_link7"
        # self.scene.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  # Removed /franka/
        #     debug_vis=False,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_link7", # Removed /franka/
        #             name="end_effector",
        #             offset=OffsetCfg(
        #                 pos=[0.0, 0.0, 0.22], 
        #             ),
        #         ),
        #     ],
        # )
        custom_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer")
        custom_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/franka_robotiq_2f_85_flattened/panda_link0",
            debug_vis=False,
            visualizer_cfg=custom_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/franka_robotiq_2f_85_flattened/panda_link{i}",
                    name=f"panda_link{i}",
                )
                for i in range(8)
            ] + [
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/franka_robotiq_2f_85_flattened/Gripper/Robotiq_2F_85/base_link",
                    name="gripper_base",
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/franka_robotiq_2f_85_flattened/Gripper/Robotiq_2F_85/base_link",
                    name="eef_frame",
                    offset=OffsetCfg(pos=[0.14, 0.0, 0.0], rot=[0.5, -0.5, 0.5, -0.5]),
                ),
            ],
        )
        #self.terminations.success=DoneTerm(func=mdp.object_stacked_upright, params={"lower_object_cfg": SceneEntityCfg("scale")})

        #self.observations.subtask_terms.appr_goal=ObsTerm(func=mdp.is_object_lifted, params={"threshold":0.15}
        #)
      


# @configclass
# class FrankaCubeEnvCfg_PLAY(FrankaDevEnvVMCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
class BinaryJointPositionZeroToOneAction(mdp.BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
@configclass
class BinaryJointPositionZeroToOneActionCfg(mdp.BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction
