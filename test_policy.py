import torch
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.torch_utils import get_torch_device

ckpt = "logs/robomimic/cube_lift_diffusion/2026-07-20_07-44-33/models/model_epoch_200.pth"
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=get_torch_device())
print("obs_normalization_stats:", getattr(policy, "obs_normalization_stats", "Not found"))
print("action_normalization_stats:", getattr(policy, "action_normalization_stats", "Not found"))
print("hasattr(policy.policy, 'obs_normalization_stats'):", hasattr(policy.policy, "obs_normalization_stats"))
if hasattr(policy.policy, "obs_normalization_stats"):
    print("policy.policy.obs_normalization_stats:", policy.policy.obs_normalization_stats)
