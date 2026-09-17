
import sys
import torch
print('PyTorch version:', torch.__version__)
from isaaclab.controllers.test_joint_pos_controller import TestJointPosController, TestJointPosControllerCfg

cfg = TestJointPosControllerCfg(command_type='relative', scale=0.5)
ctrl = TestJointPosController(cfg=cfg, num_envs=2, num_joints=7, device='cpu')

cmd = torch.ones(2, 7) * 0.1
q_curr = torch.zeros(2, 7)
ctrl.set_command(cmd, current_joint_pos=q_curr)
target = ctrl.compute()
print('Target joint pos (relative mode):', target)
assert torch.allclose(target, torch.ones(2, 7) * 0.05), 'Calculation mismatch!'

cfg_abs = TestJointPosControllerCfg(command_type='absolute', scale=1.0)
ctrl_abs = TestJointPosController(cfg=cfg_abs, num_envs=2, num_joints=7, device='cpu')
ctrl_abs.set_command(torch.full((2, 7), 0.3))
target_abs = ctrl_abs.compute()
print('Target joint pos (absolute mode):', target_abs)
assert torch.allclose(target_abs, torch.full((2, 7), 0.3)), 'Absolute mode mismatch!'
print('ALL CONTROLLER TESTS PASSED SUCCESSFULLY!')
