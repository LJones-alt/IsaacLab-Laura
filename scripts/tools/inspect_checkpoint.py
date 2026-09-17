import torch
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.torch_utils import get_torch_device
import sys

# Find a checkpoint in logs/
import os
import glob
ckpts = glob.glob("logs/**/*.pth", recursive=True)
if not ckpts:
    print("No checkpoints found.")
    sys.exit(0)

# Sort by modification time to get the latest
ckpt = max(ckpts, key=os.path.getmtime)
print(f"Loading checkpoint: {ckpt}")

try:
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=get_torch_device())
    print("\nHas action_normalization_stats in policy:", hasattr(policy, "action_normalization_stats"))
    if hasattr(policy, "action_normalization_stats"):
        print(policy.action_normalization_stats)
    
    print("\nHas obs_normalization_stats in policy.policy:", hasattr(policy.policy, "obs_normalization_stats"))
    if hasattr(policy.policy, "obs_normalization_stats"):
        stats = policy.policy.obs_normalization_stats
        if stats:
            print("Keys:", list(stats.keys()))
            for k, v in stats.items():
                print(f"  {k}: min_shape={v.get('min', []).shape if hasattr(v.get('min'), 'shape') else 'NA'} max_shape={v.get('max', []).shape if hasattr(v.get('max'), 'shape') else 'NA'}")
        else:
            print("Empty map")
except Exception as e:
    print("Error:", e)
