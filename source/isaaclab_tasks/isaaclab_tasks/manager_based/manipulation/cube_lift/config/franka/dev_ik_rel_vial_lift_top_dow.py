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

from isaaclab.managers import ObservationGroupCfg as ObsGroup

## add some cameras in
@configclass
class EventCfg():
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # randomise_object_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     # params={
    #     #     "scale_range": {"x": (0.4, 0.7), "y": (0.4, 0.7), "z": (0.5, 0.7)},
    #     #     "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     # },
    #     params={
    #         "scale_range": {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},
    #         "asset_cfg": SceneEntityCfg("object", body_names="object"),
    #     },
    # #     params={
    # #         "scale_range": {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},
    # #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    # #     },
    # )
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
            "pose_range": {"x": (0, 0.2), "y": (0, 0.25), "z": (0.02, 0.02)},
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
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        # cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
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
        #### for cosmos add the segmentation, normals and depth data
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

        # for teleop
        abs_joint_pos = ObsTerm(func=mdp.get_joint_pos)

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
        appr_goal=ObsTerm(
            func=mdp.is_object_lifted,
            params={
                "threshold":0.15
            },
        )
       

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaDevEnvCfg(dev_env_cfg.FrankaDevEnvCfg):
    def __post_init__(self):
        print("\n" + "="*50 + "\n[DEBUG] LOADED dev_ik_rel_vial_lift_top_dow.py\n" + "="*50 + "\n")
        self.observations: ObservationsCfg = ObservationsCfg()
        self.events: EventCfg = EventCfg()
        # Evaluation settings - maybe fix these ? 
        self.eval_mode = False
        self.eval_type = None
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].visible = False
        # post init of parent
        super().__post_init__()
        self.events: EventCfg = EventCfg()
        import carb
        from isaacsim.core.utils.carb import set_carb_setting
        # Force Path Tracing via Carb
        set_carb_setting(carb.settings.get_settings(), "/rtx/renderMode", "PathTracing")
        set_carb_setting(carb.settings.get_settings(), "/rtx/pathtracing/spp", 1)
        set_carb_setting(carb.settings.get_settings(), "/rtx/pathtracing/totalSpp", 1)
        
        carb_setting = carb.settings.get_settings()
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
        # put the vial on the scale
        glassware = ChemistryGlassware()
        self.scene.stirplate = glassware.stirplate(pos=[0.5, 0.0, 0.01])
        #self.scene.scale = glassware.scale(pos=[0.3, -0.3, 0.01])
        self.scene.scale = glassware.scale(pos=[0.3, -0.3, 0.0])
        self.scene.object = glassware.capped_vial(pos=[0.5, 0.0, 0.01], scale =1.0, name="object")
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=96,
            width=96,
            data_types=["rgb", "semantic_segmentation", "normals", "distance_to_image_plane"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            ### Where is this in the scene?
            offset=CameraCfg.OffsetCfg(
                pos=(1.4, 0.0, 0.5), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
            ),
        )
        self.rerender_on_reset = True
        self.sim.render_settings = {
            "rtx/renderMode": "PathTracing",
            "rtx/pathtracing/spp": 1,
            "rtx/pathtracing/totalSpp": 1,
            "rtx/pathtracing/maxBounces": 4,
            "rtx/hydra/enabled": True,
        }
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]

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
