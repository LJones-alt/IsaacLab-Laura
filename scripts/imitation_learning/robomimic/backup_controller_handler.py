from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, quat_mul
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from backup_controller import BackupControllerSM 
from backup_controller_place import BackupControllerPlaceSM
from backup_controller_insert import BackupControllerInsertSM
import torch


class BackupController:
    def __init__(self,env, device, tasktype="lift"):
        self.env = env
        self.sim = env.unwrapped.sim
        self.num_envs = env.num_envs
        self.device = device
        self.tasktype = tasktype
        self.rest_pos = self._get_rest_pos()
        self.goal_quat = self._get_goal_pos()
        self.object_goal_pos = self._get_goal_pos()
        self.object_goal_rot = self._get_goal_rot()
        self.object_start_pose = self._get_start_pos()
        self.backup_controller = self._get_controller()#BackupControllerSM(0.01*2, self.num_envs,'cuda' , 0.02)
        self.robot = env.unwrapped.scene["robot"]
        self.robot_entity_cfg , self.ee_jacobi_idx = self._setup_robot()
        self.state_guess = 0
        self.pos_gain = 0.5
        self.clamp = 0.05
        self.rot_gain = 1.0
        self.rot_clamp = 0.5
       
        
        self.reset()

    def _get_controller(self):
        match self.tasktype:
            case "lift":
                return BackupControllerSM(0.01*2, self.num_envs,'cuda' , 0.02)
            case "place":
                print(f"[DEBUG] Using Place Backup Controller")
                return BackupControllerPlaceSM(dt=0.01*2, num_envs=self.num_envs, position_threshold=0.02, device='cuda', offset = torch.tensor([[-0.1, 0, 0.1]], device='cuda:0'), goal_quat = self.object_goal_rot, goal_pos= self.object_goal_pos)
            case "insert":
                return BackupControllerInsertSM(0.01*2, self.num_envs,'cuda' , 0.02)
            case _:
                raise ValueError(f"Unknown task type {self.tasktype} for backup controller")

    def _get_goal_pos(self):
        match self.tasktype:
            case "lift":
                goal_rot = self.env.unwrapped.command_manager.get_command("object_pose")
              #  print(f"[DEBUG] Getting goal quat for lift task {goal_rot}")
                return self.env.unwrapped.command_manager.get_command("object_pose")
            case "place":
                goal_pos = self.env.unwrapped.scene["scale"].data.root_pose_w
               # print(f"[DEBUG] Getting goal pos for place task {goal_pos}")
                return self.env.unwrapped.scene["scale"].data.root_pose_w

            case "insert":
                return self.env.unwrapped.scene["vialrack"].data.root_pose_w

    def _get_goal_rot(self):
         match self.tasktype:
            case "lift":
                goal_rot = self.env.unwrapped.command_manager.get_command("object_pose")
                print(f"[DEBUG] Getting goal quat for lift task {goal_rot}")
                return self.env.unwrapped.command_manager.get_command("object_pose")
            case "place":
                goal_rot = self.env.unwrapped.command_manager.get_command("object_pose")
                print(f"[DEBUG] Getting goal quat for place task {goal_rot}")
                return self.env.unwrapped.command_manager.get_command("object_pose")

            case "insert":
                return self.env.unwrapped.scene["vialrack"].data.root_pose_w

    def reset(self):
        self.backup_controller.reset_idx()

    def _get_start_pos(self):
        object_pose_w = self.env.unwrapped.scene["object"].data.root_pose_w
        return object_pose_w

   

    def _get_rest_pos(self):
        rest_pos = torch.tensor([[0.5206, 0.0096, 0.3751]], device=self.device)
        ee_recovery_rot = torch.tensor([[ 0.6664,  0.0360,  0.7414, -0.0705]], device=self.device)
        print(f"recovery rot : {ee_recovery_rot}")
        # lets change this into the 6 element tensor they are expecting 
        roll,pitch,yaw = euler_xyz_from_quat(ee_recovery_rot)
        rest_pos = torch.cat([rest_pos, ee_recovery_rot], dim =-1)
        print(f"Rest pos {rest_pos}")
        print(f"rpy : {roll}, {pitch}, {yaw}")
        #print(f"rpy version :{torch.cat([rest_pos,roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0)], dim =-1)}")
        return rest_pos
    
    def _setup_robot(self):
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg.resolve(self.env.unwrapped.scene)
        if self.robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        return robot_entity_cfg, ee_jacobi_idx

    def get_controller_action(self, state_guess, env_num):
        
        current_object_pose = self.env.unwrapped.scene["object"].data.root_pose_w
        root_pose_w = self.robot.data.root_pose_w
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        # current EE pose in base/root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        #object_pose = self.env.unwrapped.command_manager.get_command("object_pose")
        ee_recovery, gripper , state_guess= self.backup_controller.compute(ee_pose =ee_pose_w, start_object_pose = self.object_start_pose, current_object_pose=current_object_pose, final_object_pose=self.object_goal_pos, rest_pose=self.rest_pos, sm_state=state_guess)
        #print(f"[BAKCUP] state {state_guess}")
        
        #return ee_recovery , gripper , state_guess
        ee_recovery_pos, ee_recovery_rot = ee_recovery[:,0:3], ee_recovery[:,3:7]
        target_pos_w = ee_recovery_pos.to(self.device)
        target_quat_w = ee_recovery_rot.to(self.device)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos_w, target_quat_w
        )
        ## calc the delta pos action
        delta_pos_abs = target_pos_b - ee_pos_b
        delta_pos = self.pos_gain * delta_pos_abs
        delta_pos = torch.clamp(delta_pos, min=-self.clamp, max=self.clamp)

        ## calc the delta rot action
        # compute conjugate of current quaternion (w, -x, -y, -z)
        q_cur_conj = torch.cat([ee_quat_b[:, :1], -ee_quat_b[:, 1:]], dim=-1)
        # compute relative quaternion q_rel = q_target * q_cur_conj
        delta_q = quat_mul(target_quat_b, q_cur_conj)  # uses imported quat_mul
        # convert delta quaternion to RPY (XYZ extrinsic)
        r, p, y = euler_xyz_from_quat(delta_q, wrap_to_2pi=False)  # each shape (N,)
        delta_rpy = torch.stack([r, p, y], dim=-1)  # [N,3]
        delta_rpy = self.rot_gain * delta_rpy
        delta_rpy = torch.clamp(delta_rpy, min=-self.rot_clamp, max=self.rot_clamp)
        # --- build controller command: [dx,dy,dz, droll,dpitch,dyaw] ---
        ee_goal = torch.cat([delta_pos, delta_rpy], dim=-1)  # [N,6]

        action=torch.cat([ee_goal, gripper.unsqueeze(0)], dim=-1)
        return action, state_guess