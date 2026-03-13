import warp as wp
import gymnasium as gym
import torch
from collections.abc import Sequence
import numpy as np

# this is a backup state machine controller
# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class BackupSMWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    APPROACH_ABOVE_OBJECT = wp.constant(0.3)
    APPROACH_OBJECT = wp.constant(0.3)
    GRASP_OBJECT = wp.constant(0.1)  # 1 second wait + 0.3 for gripper to close
    LIFT_OBJECT = wp.constant(0.3)
    MIDPOINT = wp.constant(0.1)
    APPROACH_ABOVE_GOAL = wp.constant(0.1)
    APPROACH_GOAL = wp.constant(6)

class BackupSM:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    MIDPOINT = wp.constant(5)
    APPROACH_ABOVE_GOAL = wp.constant(6)
    APPROACH_GOAL = wp.constant(7)
    #UNGRASP_OBJECT = wp.constant(7)

@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform), # current ee pose
    object_pose: wp.array(dtype=wp.transform), # start object location
    current_object_pose: wp.array(dtype=wp.transform), # current object pose 
    final_object_pose: wp.array(dtype=wp.transform),
    rest_ee_pose : wp.array(dtype=wp.transform), # rest position
    des_ee_pose: wp.array(dtype=wp.transform), # what we return 
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
    debug_des_pose: wp.array(dtype=wp.transform),   # NEW
    debug_cur_pose: wp.array(dtype=wp.transform),   # NEW
    debug_state: wp.array(dtype=int),               # New
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == BackupSM.REST:
        # First lift up to a safe height, then go to rest position
        current_ee_pos = wp.transform_get_translation(ee_pose[tid])
        rest_pos = wp.transform_get_translation(rest_ee_pose[tid])
        rest_rot = wp.transform_get_rotation(rest_ee_pose[tid])
        
        # If we're below safe height, first move up
        safe_height = 0.15  # Safe z-height to avoid collisions
        if current_ee_pos.z < safe_height:
            # Lift straight up first
            lift_pos = wp.vec3(current_ee_pos.x, current_ee_pos.y, safe_height)
            des_ee_pose[tid] = wp.transform(lift_pos, rest_rot)
        else:
            # Already at safe height, move to rest position
            des_ee_pose[tid] = rest_ee_pose[tid]
        
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(rest_ee_pose[tid]),
            0.02,
        ):
            if sm_wait_time[tid] >= BackupSMWaitTime.REST:
                print("[SM_INFO] : Moving from REST to APPROACH_ABOVE_OBJECT")
                # move to next state and reset wait time
                sm_state[tid] = BackupSM.APPROACH_ABOVE_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == BackupSM.APPROACH_ABOVE_OBJECT:
       # print("[SM] APPR above obj")
        pose_pos = wp.transform_get_translation(current_object_pose[tid])
        pose_rot = wp.transform_get_rotation(rest_ee_pose[tid])
        # Go to a safe height ABOVE the object (z + 0.15m) with x offset
        safe_above_pos = wp.vec3(pose_pos.x - 0.1, pose_pos.y, pose_pos.z + 0.15)
        des_ee_pose[tid] = wp.transform(safe_above_pos, pose_rot)
        gripper_state[tid] = GripperState.OPEN  # Open gripper in preparation
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= BackupSMWaitTime.APPROACH_ABOVE_OBJECT:
                # move to next state and reset wait time
                print("[SM_INFO] : Moving from APPR_ABOVE to APPROACH_OBJECT")
                sm_state[tid] = BackupSM.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == BackupSM.APPROACH_OBJECT:
       # print("[SM] approach object")
        
        pose_pos = wp.transform_get_translation(current_object_pose[tid])
        pose_rot = wp.transform_get_rotation(rest_ee_pose[tid])
        # Apply offset in x-direction (5 cm = 0.05 m)
        offset_pos = wp.vec3(pose_pos.x -0.1, pose_pos.y+0.02, pose_pos.z+0.02) #sample vial + 0.04
        des_ee_pose[tid] = wp.transform(offset_pos, pose_rot)
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= BackupSMWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                print("[SM_INFO] : Moving from APPR_OBJ to GRASP_OBJECT")
                sm_state[tid] = BackupSM.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == BackupSM.GRASP_OBJECT:
       # print("[SM] GRASP")
        pose_pos = wp.transform_get_translation(current_object_pose[tid])
        pose_rot = wp.transform_get_rotation(rest_ee_pose[tid])
        # Stay at the approach position while closing gripper
        offset_pos = wp.vec3(pose_pos.x - 0.1, pose_pos.y + 0.02, pose_pos.z + 0.02)
        des_ee_pose[tid] = wp.transform(offset_pos, pose_rot)
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= BackupSMWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            print("[SM_INFO] : Moving from GRSP to LIFT_OBJECT")
            sm_state[tid] = BackupSM.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
            
    elif state == BackupSM.LIFT_OBJECT:
        #print("[SM] LIFT") # uses starting position
        pose_pos = wp.transform_get_translation(object_pose[tid])
        pose_rot = wp.transform_get_rotation(rest_ee_pose[tid])
        # Apply offset in x-direction (5 cm = 0.05 m)
        offset_pos = wp.vec3(pose_pos.x -0.1, pose_pos.y+0.02, pose_pos.z+0.2) #sample vial + 0.04
        des_ee_pose[tid] = wp.transform(offset_pos, pose_rot)
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            0.03,
        ):
            # wait for a while
            if sm_wait_time[tid] >= BackupSMWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                print("[SM_INFO] : Moving from LIFT to MIDPOINT")
                sm_state[tid] = BackupSM.MIDPOINT
                sm_wait_time[tid] = 0.0
    elif state == BackupSM.MIDPOINT:
        #print("[SM] MIDPOINT")
        pose_pos = wp.transform_get_translation(rest_ee_pose[tid])
        pose_rot = wp.transform_get_rotation(rest_ee_pose[tid])
        # Apply offset in x-direction (5 cm = 0.05 m)
        #offset_pos = wp.vec3(pose_pos.x -0.1, pose_pos.y+0.02, pose_pos.z+0.2) #sample vial + 0.04
        des_ee_pose[tid] = rest_ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            0.03,
        ):
            # wait for a while
            if sm_wait_time[tid] >= BackupSMWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                print("[SM_INFO] : Moving from MID to APPR goal")
                sm_state[tid] = BackupSM.APPROACH_ABOVE_GOAL
                sm_wait_time[tid] = 0.0
    elif state == BackupSM.APPROACH_ABOVE_GOAL:
        #print("[SM] Approach above goal")
        offset_pos = wp.transform_get_translation(offset[tid])
        offset_rot = wp.transform_get_rotation(offset[tid])
        offset_pos = wp.vec3(offset_pos.x, offset_pos.y, offset_pos.z)  # raise 25 cm 
        new_offset = wp.transform(offset_pos, offset_rot)
        above_target_pose = wp.transform_multiply(new_offset, final_object_pose[tid])
        # Blend time for smooth approach
        APPROACH_BLEND_TIME = 0.4  # seconds, tune as needed
        alpha = wp.clamp(sm_wait_time[tid] / APPROACH_BLEND_TIME, 0.0, 1.0)
        # Interpolate position and rotation
        current_pos = wp.transform_get_translation(ee_pose[tid])
        current_rot = wp.transform_get_rotation(ee_pose[tid])

        target_pos = wp.transform_get_translation(above_target_pose)
        target_rot = wp.transform_get_rotation(above_target_pose)

        pos_interp = wp.lerp(current_pos, target_pos, alpha)
        rot_interp = wp.quat_slerp(current_rot, target_rot, alpha)
        # Set interpolated pose
        des_ee_pose[tid] = wp.transform(pos_interp, rot_interp)
        gripper_state[tid] = GripperState.CLOSE
        # Evaluate readiness to transition
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), 
            0.02):
            if sm_wait_time[tid] >= BackupSMWaitTime.APPROACH_ABOVE_GOAL:
                print("[SM_INFO] : Moving from APPR_ABOVE to APPROACH_GOAL")
                sm_state[tid] = BackupSM.APPROACH_GOAL
                sm_wait_time[tid] = 0.0
    elif state == BackupSM.APPROACH_GOAL:
        #print("[SM] approach goal")
       # print("[SM_INFO] : in state ungrasp")
        pose_pos = wp.transform_get_translation(final_object_pose[tid])
        pose_rot = wp.transform_get_rotation(final_object_pose[tid])
        # Apply offset in x-direction (5 cm = 0.05 m)
        offset_pos = wp.vec3(pose_pos.x, pose_pos.y+0.1, pose_pos.z+0.01)
        des_ee_pose[tid] = wp.transform(offset_pos, pose_rot)
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            0.01,
        ):
            if sm_wait_time[tid] >= BackupSMWaitTime.APPROACH_GOAL:
                # move to next state and reset wait time
                print("[SM_INFO] : Moving from apprach goal to rest")
                sm_state[tid] = BackupSM.REST
                sm_wait_time[tid] = 0.0
            
        
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
    debug_des_pose[tid] = des_ee_pose[tid]
    debug_cur_pose[tid] = ee_pose[tid]
    debug_state[tid] = sm_state[tid]



class BackupControllerSM:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.02, offset = torch.tensor([[-0.1, 0, 0.1]], device='cuda:0')):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        ## debug

        print(f'dt : {dt},num_envs {num_envs}, device {device}, pos_thresh {position_threshold} ')
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        self.final_object_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.rest_pose = torch.zeros((self.num_envs, 7), device=self.device)
        # approach above object offset
        
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.2
        self.offset[:,0] = -0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.final_object_pose_wp = wp.from_torch(self.final_object_pose, wp.transform)
        self.rest_pose_wp = wp.from_torch(self.rest_pose, wp.transform)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
         ## For Debug 
        self.debug_des_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.debug_cur_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.debug_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)

        self.debug_des_pose_wp = wp.from_torch(self.debug_des_pose, wp.transform)
        self.debug_cur_pose_wp = wp.from_torch(self.debug_cur_pose, wp.transform)
        self.debug_state_wp = wp.from_torch(self.debug_state, wp.int32)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        print(f'[SM RESET]')
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def _ensure_7d(self,tensor: torch.Tensor, device):
        # accepts (N,3) -> returns (N,7) with identity quaternion
        if tensor.ndim != 2:
            raise ValueError(f"expected 2D tensor, got shape {tensor.shape}")
        if tensor.shape[1] == 7:
            return tensor
        if tensor.shape[1] == 3:
            quat = torch.tensor([0.0,0.0,0.0,1.0], device=device, dtype=tensor.dtype)
            quat = quat.unsqueeze(0).expand(tensor.shape[0], -1)
            return torch.cat([tensor, quat], dim=-1)
        raise ValueError(f"pose must have 3 or 7 elements, got {tensor.shape[1]}")


    def compute(self, ee_pose: torch.Tensor, start_object_pose: torch.Tensor, current_object_pose: torch.Tensor, final_object_pose: torch.Tensor, rest_pose: torch.Tensor, sm_state: int =0, env_num = 0) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        # print(f'[SM DEBUG] Doing State Comp')
        # print(f'[SM DEBUG] ee_pose  {ee_pose}')
        # print(f'[SM DEBUG] object pose {object_pose}')
        # print(f'[SM DEBUG] desired pose {des_object_pose}')
        # print(f'[SM DEBUG] final pose {final_object_pose}')
        # print(f'[SM DEBUG] rest pose {rest_pose}')
        # before reordering
        # print("\n[BackupControllerSM::compute] called with inputs:")
        # print(f"  sm_state: {sm_state}, env_num: {env_num}")

        # def short_tensor(t, name):
        #     if t is None:
        #         print(f"  {name}: None")
        #     elif not isinstance(t, torch.Tensor):
        #         print(f"  {name}: {t} (type={type(t)})")
        #     else:
        #         shape = tuple(t.shape)
        #         device = t.device
        #         dtype = t.dtype
        #         # show first few elements flattened
        #         vals = t.flatten()[:6].detach().cpu().numpy()
        #         print(f"  {name}: shape={shape}, device={device}, dtype={dtype}, values={vals}")

        # short_tensor(ee_pose, "ee_pose")
        # short_tensor(object_pose, "object_pose")
        # short_tensor(des_object_pose, "des_object_pose")
        # short_tensor(final_object_pose, "final_object_pose")
        # short_tensor(rest_pose, "rest_pose")
        # print("-" * 80)
        # print(f"Got given state {sm_state}")
        current_object_pose = self._ensure_7d(current_object_pose, self.device)
        final_object_pose = self._ensure_7d(final_object_pose, self.device)
        ee_pose = self._ensure_7d(ee_pose, self.device)
        start_object_pose = self._ensure_7d(start_object_pose, self.device)
        rest_pose = self._ensure_7d(rest_pose, self.device)
        # now safe to reorder
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        start_object_pose = start_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        current_object_pose = current_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        final_object_pose = final_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        rest_pose = rest_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        start_object_pose_wp = wp.from_torch(start_object_pose.contiguous(), wp.transform)
        current_object_pose_wp = wp.from_torch(current_object_pose.contiguous(), wp.transform)
        final_object_pose_wp = wp.from_torch(final_object_pose.contiguous(), wp.transform)
        rest_pose_wp = wp.from_torch(rest_pose.contiguous(), wp.transform)
        # run state machine
        # get state from env - convert to warp again
        self.sm_state[env_num] = sm_state
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        # launch
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                start_object_pose_wp,
                current_object_pose_wp,
                final_object_pose_wp,
                rest_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
                self.debug_des_pose_wp,
                self.debug_cur_pose_wp,
                self.debug_state_wp,
            ],
            device=self.device,
        )
        des_debug = self.debug_des_pose.detach().cpu().numpy()
        cur_debug = self.debug_cur_pose.detach().cpu().numpy()
        debug_state = self.debug_state.detach().cpu().numpy()
        for i in range(self.num_envs):
            des_pos, des_quat = des_debug[i][:3], des_debug[i][3:]
            cur_pos, cur_quat = cur_debug[i][:3], cur_debug[i][3:]
            # compute Euclidean distance with numpy
            dist = np.linalg.norm(cur_pos - des_pos)
            # print(f"StateMachine debug state {debug_state}")
            # print(f"[Env {i}]")
            # print(f" Goal pose : {final_object_pose}")
            # print(f"   Current -> pos: {cur_pos}, quat: {cur_quat}")
            # print(f"   Desired -> pos: {des_pos}, quat: {des_quat}")
            # print(f"   Euclidean distance: {dist:.4f}")
        #print(f"returned state : {self.sm_state}")
        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        #print(f'[SM DEBUG] new desired poition : {des_ee_pose}, gripper state {self.des_gripper_state}')
        return des_ee_pose , self.des_gripper_state, self.sm_state.clone().item()
        #return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)