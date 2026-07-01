# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import numpy as np
##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
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
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=0.8,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
#FRANKA_PANDA_HIGH_PD_CFG.soft_joint_pos_limit_factor=0.8,
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""


FRANKA_ROBOTIQ_GRIPPER_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.pos = (-0.85, 0, 0.76)
FRANKA_ROBOTIQ_GRIPPER_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=1100.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=1000.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    # enable the gripper to grasp in a parallel manner
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
    # set PD to zero for passive joints in close-loop gripper
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
}

PANDA_OFFSET_ROBOTIQ_CFG = ArticulationCfg(
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path=f"source/isaaclab_assets/gripper/panda_rf85_gripper/panda_rf85_gripper.usd",
    #     activate_contact_sensors=True,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=True,
    #         max_depenetration_velocity=5.0,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=2
    #     ),
    #     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    # ),
    spawn=sim_utils.UsdFileCfg(
        #usd_path=f"source/isaaclab_assets/isaaclab_assets/robots/Collected_panda_robotiq_v4/panda_robotiq_v4.usd",
        #usd_path=f"source/isaaclab_assets/isaaclab_assets/robots/new_franka_robotiq_test.usd",
        usd_path=f"docs/franka_robotiq_2f_85_flattened.usd" ,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=24, 
            solver_velocity_iteration_count=8
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "finger_joint": 0.0,
            ".*_inner_finger_joint": 0.0,
           # ".*_inner_finger_pad_joint": 0.0,
            #
            # ".*_outer_.*_joint": 0.0,
            ".*_inner_finger_knuckle_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,

        },
    ),
    
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=3.14,
            stiffness=800.0,
            damping=80.0,
        ),
        "gripper_drive": ImplicitActuatorCfg(
            # Drive ONLY the single primary joint explicitly
            joint_names_expr=["finger_joint"], 
            effort_limit_sim=1000.0, # Lower this! 1650 is way too high and introduces unstable energy
            velocity_limit_sim=1.0, # Mimic real Robotiq speeds to keep physics stable
            stiffness=1000.0, 
            damping=10.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            # Exclude finger_joint cleanly by being explicit or using strict prefixes
            # joint_names_expr=[
            #     ".*_inner_finger_joint",
            #     #".*_inner_finger_pad_joint",
            #     "right_outer_knuckle_joint",
            #     ".*_outer_finger_joint"
            # ],
            joint_names_expr=[
                ".*_inner_finger_joint",
                ".*_inner_finger_knuckle_joint", # Updated name
                "right_outer_knuckle_joint"      # Explicitly named
            ],
            effort_limit_sim=0.0,
            velocity_limit_sim=1.0,
            stiffness=0.0, # Must be 0! Setting stiffness > 0 fights the physical mimic loop loops
            damping=0.1,   # Add a little damping to absorb noise
        ),
        # "gripper_drive": ImplicitActuatorCfg(
        #     # Drive the primary joint
        #     joint_names_expr=["finger_joint"], 
        #     effort_limit_sim=1650,
        #     velocity_limit_sim=1.0,
        #     stiffness=2000.0, # Increased to ensure it actually closes under load
        #     damping=2.0,
        # ),
        # "gripper_passive": ImplicitActuatorCfg(
        #     # Group ALL other gripper mechanism joints to be passive (stiffness=0)
        #     # This allows the physical/loop constraints to maintain parallelism
        #     joint_names_expr=[
        #         "right_outer_knuckle_joint",
        #         ".*_outer_finger_joint",
        #         ".*_inner_finger_joint",
        #         ".*_inner_finger_pad_joint"
        #     ],
        #     effort_limit_sim=0.0,
        #     velocity_limit_sim=1.0,
        #     stiffness=10.0,
        #     damping=0.1,
        # ),
    },
    soft_joint_pos_limit_factor=0.8,
)

TEST= ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"source/isaaclab_assets/isaaclab_assets/robots/franka_robotiq_2f85_orient_variant.usd",
            #usd_path= (f"docs/franka_robotiq_2f_85_flattened.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0, 0, 0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * np.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * np.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * np.pi,
                "panda_joint7": 0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                # stiffness=None,
                # damping=None,
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                # stiffness=None,
                # damping=None,
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=None,
                damping=None,
                # effort_limit=150.0,
                velocity_limit=5.0, #2.175,
                # stiffness=1000.0,
                # damping=40.0,
            ),
        },
    )

    

    # Per-link frame visualization for debugging. EE pose still comes from articulation
    # body state (faster); this sensor is purely for debug rendering. Flip debug_vis to
    # True to render RGB axes at every tracked link in the viewport.
    


