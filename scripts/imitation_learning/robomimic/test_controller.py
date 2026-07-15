from enum import Enum
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, quat_mul, axis_angle_from_quat
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

class TaskState(Enum):
    REST = 0
    APPROACH_ABOVE_OBJECT = 1
    APPROACH_OBJECT = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4
    MIDPOINT = 5
    APPROACH_ABOVE_GOAL = 6
    APPROACH_GOAL = 7
    UNGRASP_OBJECT = 8

class GripperState:
    """States for the gripper."""
    OPEN = 0.0
    CLOSE = 1.0

class TestController:
    def __init__(self, device, env):
        self.device = device
        self.env = env
        self.object_start_pos = torch.tensor([0, 0, 0], device=self.device)
        self.object_goal_pos = torch.tensor([0, 0, 0], device=self.device)
        self.object_start_rot = torch.tensor([0, 0, 0, 0], device=self.device)
        self.object_goal_rot = torch.tensor([0, 0, 0, 0], device=self.device)
        self.current_object_pos=torch.tensor([0, 0, 0], device=self.device)
        self.current_ee_pos = torch.tensor([0, 0, 0], device=self.device)
        self.current_ee_rot = torch.tensor([0, 0, 0, 0], device=self.device)
        self.rest_pos = torch.tensor([0.5206, 0.0096, 0.3751], device=self.device)
        self.rest_rot = torch.tensor([[ 0.4821,  0.4953, -0.5114, -0.5106]], device=self.device)
        self.ee_offset= torch.tensor([0, 0, 0.018], device=self.device)
        self.robot = env.unwrapped.scene["robot"]
        self.root_pose_w = self.robot.data.root_pose_w # world root pos
        self.state=TaskState.REST
        self.gripper_state=GripperState.OPEN
        self.desired_pose=torch.tensor([0, 0, 0, 0, 0, 0, 0], device=self.device)
        self.threshold = 0.05
        self.clamp = 0.05
        self.state_timer = 50
        self.state_timer_reset = 50
        self.offset_pos = torch.tensor([[0.14, 0.0, 0.0]], device=self.device)
        self.offset_quat = torch.tensor([[0.5, -0.5, 0.5, -0.5]], device=self.device)
        self.robot_entity_cfg , self.ee_jacobi_idx = self._setup_robot()
        self._init_task()

    def _init_task(self):
        # get object pos and quat in world frame
        obj_pos_w = self.env.unwrapped.scene["object"].data.root_pose_w[:, :3]
        obj_quat_w = self.env.unwrapped.scene["object"].data.root_pose_w[:, 3:7]
        # convert to robot base frame
        obj_pos_b, _ = subtract_frame_transforms(
            self.root_pose_w[:, :3], self.root_pose_w[:, 3:7],
            obj_pos_w, obj_quat_w
        )
        self.object_start_pos = obj_pos_b[0, :3]
        
        # get goal (scale) pos and quat in world frame
        goal_pos_w = self.env.unwrapped.scene["scale"].data.root_pose_w[:, :3]
        goal_quat_w = self.env.unwrapped.scene["scale"].data.root_pose_w[:, 3:7]
        # convert to robot base frame
        goal_pos_b, _ = subtract_frame_transforms(
            self.root_pose_w[:, :3], self.root_pose_w[:, 3:7],
            goal_pos_w, goal_quat_w
        )
        self.object_goal_pos = goal_pos_b[0, :3]
        
        self.object_start_rot = self.rest_rot
        self.object_goal_rot = self.rest_rot
        print("[TEST_CONTROLLER] INIT")
        print(f"object start pos (base frame): {self.object_start_pos}")
        print(f"object goal pos (base frame): {self.object_goal_pos}")
        print(f"object start rot: {self.object_start_rot}")
        print(f"object goal rot: {self.object_goal_rot}")
        self._update_ee_pose()

    def _update_ee_pose(self):
        # find relative ee pose and rot 
        raw_link_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        link_pos_w = raw_link_pose_w[:, 0:3]
        link_quat_w = raw_link_pose_w[:, 3:7]
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            link_pos_w, link_quat_w,
            self.offset_pos, self.offset_quat
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            self.root_pose_w[:, 0:3], self.root_pose_w[:, 3:7],
            ee_pos_w, ee_quat_w
        )
        self.current_ee_pos = ee_pos_b
        self.current_ee_rot = ee_quat_b

       # print(f"[TEST_CONTROLLER] current ee pos: {self.current_ee_pos}")
        #print(f"[TEST_CONTROLLER] current ee rot: {self.current_ee_rot}")
    
    def _setup_robot(self):
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["base_link"])
        robot_entity_cfg.resolve(self.env.unwrapped.scene)
        if self.robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        return robot_entity_cfg, ee_jacobi_idx

    # private internal
    def _get_goal_pose(self):
        match self.state:
            case TaskState.REST:
                self.gripper_state=GripperState.OPEN
                self._update_ee_pose()
                if self.current_ee_pos[0, 2].item() < 0.2:
                    # if below 0.2, then move straight uppwards only
                    pos = torch.add(self.current_ee_pos, torch.tensor([0, 0, 0.2], device=self.device))
                    return pos, self.rest_rot
                return self.rest_pos, self.rest_rot
            case TaskState.APPROACH_ABOVE_OBJECT:
                self.gripper_state=GripperState.OPEN
                position= torch.add(self.object_start_pos, torch.tensor([0, 0, 0.1], device=self.device))
                return position, self.rest_rot
            case TaskState.APPROACH_OBJECT:
                self.gripper_state=GripperState.OPEN
                position= torch.add(self.object_start_pos, self.ee_offset)
                return position, self.rest_rot
            case TaskState.GRASP_OBJECT:
                self.gripper_state=GripperState.CLOSE
                position= torch.add(self.object_start_pos, self.ee_offset)
                return position, self.rest_rot
            case TaskState.LIFT_OBJECT:
                self.gripper_state=GripperState.CLOSE
                position= torch.add(self.object_start_pos, torch.tensor([0, 0, 0.1], device=self.device))
                return position, self.rest_rot
            case TaskState.MIDPOINT:
                self.gripper_state=GripperState.CLOSE
                position= torch.add(self.object_start_pos, torch.tensor([0, 0, 0.2], device=self.device))
                return position, self.rest_rot
            case TaskState.APPROACH_ABOVE_GOAL:
                self.gripper_state=GripperState.CLOSE
                position= torch.add(self.object_goal_pos, torch.tensor([0, 0, 0.3], device=self.device))
                return position, self.rest_rot
            case TaskState.APPROACH_GOAL:
                self.gripper_state=GripperState.CLOSE
                position= torch.add(self.object_goal_pos, torch.tensor([0, 0, 0.12], device=self.device))
                return position, self.rest_rot
            case TaskState.UNGRASP_OBJECT:
                self.gripper_state=GripperState.OPEN
                position= torch.add(self.object_goal_pos, torch.tensor([0, 0, 0.1], device=self.device))
                return position, self.rest_rot

    def _calc_action(self, desired_pos, desired_rot):
        delta_pos = desired_pos - self.current_ee_pos
        delta_pos = torch.clamp(delta_pos, min=-self.clamp, max=self.clamp)
        #print(f"[TEST_CONTROLLER] DELTA POS: {delta_pos}")
        # conjugate of rot (w, -x, -y, -z)
        #q_cur_conj = torch.cat([self.current_ee_rot, -self.current_ee_rot], dim=-1)
        q_cur_conj = self.current_ee_rot.clone()
        q_cur_conj[..., 1:] *= -1 # Assuming (w, x, y, z).
        #print(f"[TEST_CONTROLLER] Q CUR CONJ: {q_cur_conj}")
        delta_q = quat_mul(desired_rot, q_cur_conj)
        #print(f"[TEST_CONTROLLER] DELTA Q: {delta_q}")
        delta_axis_angle = axis_angle_from_quat(delta_q)
        #print(f"[TEST_CONTROLLER] DELTA AXIS ANGLE: {delta_axis_angle}")
        delta_rot_action = 1.0 * delta_axis_angle
        #print(f"[TEST_CONTROLLER] DELTA ROT ACTION: {delta_rot_action}")
        delta_rot_action = torch.clamp(delta_rot_action, min=-0.5, max=0.5)
        #print(f"[TEST_CONTROLLER] DELTA ROT ACTION CLAMPED: {delta_rot_action}")
        ee_goal = torch.cat([delta_pos, delta_rot_action], dim=-1)
        #print(f"[TEST_CONTROLLER] EE GOAL: {ee_goal}")
        gripper_action = torch.tensor([[self.gripper_state]], device=self.device)
        self.desired_pose=torch.cat([ee_goal, gripper_action], dim=-1)
        #print(f"[TEST_CONTROLLER] ACTION: {self.desired_pose}")
        return self.desired_pose

            
    def get_action(self):
        # quickly update the current ee pose and rot
        self._update_ee_pose()
        desired_pos, desired_rot = self._get_goal_pose()
        match self.state:
            case TaskState.REST:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    if self.state_timer<self.state_timer_reset:# if below thresold, then move on to new state
                        print("[TEST_CONTROLLER] REST -> APPROACH_ABOVE_OBJECT")
                        self.state = TaskState.APPROACH_ABOVE_OBJECT
                        self.state_timer=self.state_timer_reset
                        #self.state=TaskState.REST
            case TaskState.APPROACH_ABOVE_OBJECT:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    # if below thresold, then move on to new state
                    if self.state_timer<self.state_timer_reset:
                        print("[TEST_CONTROLLER] APPROACH_ABOVE_OBJECT -> APPROACH_OBJECT")
                        self.state = TaskState.APPROACH_OBJECT
                        self.state_timer=self.state_timer_reset
            case TaskState.APPROACH_OBJECT:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    # if below thresold, then move on to new state
                    if self.state_timer<self.state_timer_reset/2:
                        print("[TEST_CONTROLLER] APPROACH_OBJECT -> GRASP_OBJECT")
                        self.state = TaskState.GRASP_OBJECT
                        self.state_timer=self.state_timer_reset
            case TaskState.GRASP_OBJECT:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    # if below thresold, then move on to new state
                    if self.state_timer<self.state_timer_reset/2:
                        print("[TEST_CONTROLLER] GRASP_OBJECT -> LIFT_OBJECT")
                        self.state = TaskState.LIFT_OBJECT
                        self.state_timer=self.state_timer_reset    
            case TaskState.LIFT_OBJECT:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    # if below thresold, then move on to new state
                    if self.state_timer<self.state_timer_reset:
                        print("[TEST_CONTROLLER] LIFT_OBJECT -> MIDPOINT")
                        self.state = TaskState.MIDPOINT
                        self.state_timer=self.state_timer_reset
            case TaskState.MIDPOINT:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    # if below thresold, then move on to new state
                    if self.state_timer<self.state_timer_reset:
                        print("[TEST_CONTROLLER] MIDPOINT -> APPROACH_ABOVE_GOAL")
                        self.state = TaskState.APPROACH_ABOVE_GOAL
                        self.state_timer=self.state_timer_reset
            case TaskState.APPROACH_ABOVE_GOAL:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    # if below thresold, then move on to new state
                    if self.state_timer<self.state_timer_reset:
                        print("[TEST_CONTROLLER] APPROACH_ABOVE_GOAL -> APPROACH_GOAL")
                        self.state = TaskState.APPROACH_GOAL
                        self.state_timer=self.state_timer_reset
            case TaskState.APPROACH_GOAL:
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1# if below thresold, then move on to new state
                    if self.state_timer<self.state_timer_reset/2:
                        print("[TEST_CONTROLLER] APPROACH_GOAL -> UNGRASP_OBJECT")
                        self.state = TaskState.UNGRASP_OBJECT
                        self.state_timer=self.state_timer_reset
            case TaskState.UNGRASP_OBJECT:
                
                if torch.norm(self.current_ee_pos - desired_pos) < self.threshold and torch.norm(self.current_ee_rot - desired_rot) < self.threshold:
                    self.state_timer-=1
                    if self.state_timer<1:
                        # if below thresold, then move on to new state
                        print("[TEST_CONTROLLER] UNGRASP_OBJECT -> REST")
                        self.state = TaskState.REST
                        self.state_timer=self.state_timer_reset
            case _:
                print("[TEST_CONTROLLER] UNKNOWN STATE")
                self.state=TaskState.REST
    #    print(f"[TEST_CONTROLLER] State : {self.state}")
     #   print(f"[TEST_CONTROLLER] DESIRED POSE: {desired_pos}, {desired_rot}")
        action= self._calc_action(desired_pos, desired_rot)
        return action
                    
    def reset(self):
        self.state=TaskState.REST
        self.state_timer=self.state_timer_reset
        self._init_task()
        print("[TEST_CONTROLLER] RESET")