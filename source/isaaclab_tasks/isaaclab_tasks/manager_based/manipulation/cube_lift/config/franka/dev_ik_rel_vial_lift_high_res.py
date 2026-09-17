# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg#, DifferentialInverseKinematicsActionCfg_MOD
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
from isaaclab.actuators import ImplicitActuatorCfg
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.glassware.glassware import ChemistryGlassware
from isaaclab_tasks.manager_based.manipulation.cube_lift import mdp
from isaaclab_tasks.manager_based.manipulation.cube_lift.mdp import franka_stack_events

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 

# Pre-defined configs
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup


@configclass
class EventCfg():
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    reset_object_position = EventTerm(
        func=mdp.reset_place_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.4), "y": (-0.1, 0.3), "z": (0.022, 0.022)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="object"),
            "asset2_cfg": SceneEntityCfg("stirplate"),
            "asset3_cfg": SceneEntityCfg("scale")
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
        target_object_position = ObsTerm(func=mdp.target_position, params={"object_cfg": SceneEntityCfg("scale")})
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        
        # RGB Observations (Updated resolution configured in Scene)
        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )
        
        # Depth Observations
        table_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            }
        )
        # wrist_cam_depth = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("wrist_cam"),
        #         "data_type": "distance_to_image_plane",
        #         "normalize": False,
        #     }
        # )

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
        appr_goal = ObsTerm(
            func=mdp.is_object_lifted,
            params={
                "threshold": 0.15
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaDevEnvCfg(dev_env_cfg.FrankaDevEnvCfg):
    def __post_init__(self):
        print("\n" + "="*50 + "\n[DEBUG] LOADED dev_ik_rel_vial_lift_top_dow.py\n" + "="*50 + "\n")
        self.observations: ObservationsCfg = ObservationsCfg()
        self.events: EventCfg = EventCfg()
        
        # Evaluation settings
        self.eval_mode = False
        self.eval_type = None
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].visible = False
        
        # Post-init of parent
        super().__post_init__()
        self.events: EventCfg = EventCfg()
        
        import carb
        from isaacsim.core.utils.carb import set_carb_setting

        set_carb_setting(carb.settings.get_settings(), "/rtx/renderMode", "RayTracing")
        
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
        
        glassware = ChemistryGlassware()
        self.scene.table2 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table2",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0.75, 0], rot=[1.0, 0, 0, 0]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
        )
        self.scene.Table3 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table3",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, -0.75, 0], rot=[1.0, 0, 0, 0]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
        )
        self.scene.stirplate = glassware.IKAplate(pos=[0.1, 0.0, 0.01])
        self.scene.scale = glassware.scale(pos=[0.15, -0.4, 0.01])
        
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0, 0.05], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Table Camera (224x224 RGB + Depth)
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 5.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.8, 0.0, 0.5), rot=(0.445, -0.54953, -0.54953, 0.445), convention="ros"
            ),
        )
        
        # Wrist Camera (224x224 RGB + Depth)
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70106, 0.0923, 0.0923, -0.70106), convention="ros"
            ),
        )
        
        self.rerender_on_reset = True
        self.sim.render_settings = {
            "rtx/renderMode": "RayTracing",
            "rtx/hydra/enabled": True,
        }
        self.sim.render.antialiasing_mode = "DLAA"
        
        # Updated list of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam", "table_cam_depth", "wrist_cam_depth"]

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            # actuators={
            #     "panda_shoulder": ImplicitActuatorCfg(
            #         joint_names_expr=["panda_joint[1-4]"],
            #         effort_limit=87.0,
            #         velocity_limit=2.175,
            #         stiffness=400.0,   # raised from 80 — enough to fight gravity
            #         damping=40.0,      # raised from 4
            #     ),
            #     "panda_forearm": ImplicitActuatorCfg(
            #         joint_names_expr=["panda_joint[5-7]"],
            #         effort_limit=12.0,
            #         velocity_limit=2.61,
            #         stiffness=400.0,
            #         damping=40.0,
            #     ),
            # },
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.6917,
                    "panda_joint3": 0.0054,
                    "panda_joint4": -2.4073,
                    "panda_joint5": -0.0218,
                    "panda_joint6": 1.7427,
                    "panda_joint7": 0.7547,
                    "panda_finger_joint.*": 0.04,
                },
            ),
        )
        
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        # Relative position controller
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
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False