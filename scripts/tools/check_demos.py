import h5py
import numpy as np

f = h5py.File('/workspace/isaaclab/docs/place/glassware_stack_demos_split.hdf5', 'r')

# Analyze gripper action (dim 6) over normalized time
demos = list(f['data'].keys())
print(f"Number of demos: {len(demos)}")

# Get a few demo lengths
lengths = [f['data'][d]['actions'].shape[0] for d in demos[:10]]
print(f"Demo lengths (first 10): {lengths}")

# Analyze gripper at different phases
early_gripper = []  # First 10% of each demo
mid_gripper = []    # 30-50% of each demo  
late_gripper = []   # Last 20% of each demo

for demo in demos:
    actions = f['data'][demo]['actions'][:]
    n = len(actions)
    gripper = actions[:, 6]  # Gripper is dim 6
    
    early_gripper.extend(gripper[:int(n*0.1)])
    mid_gripper.extend(gripper[int(n*0.3):int(n*0.5)])
    late_gripper.extend(gripper[int(n*0.8):])

print(f"\nGripper action analysis by phase:")
print(f"Early (0-10%):  mean={np.mean(early_gripper):.3f}, std={np.std(early_gripper):.3f}")
print(f"Mid (30-50%):   mean={np.mean(mid_gripper):.3f}, std={np.std(mid_gripper):.3f}")
print(f"Late (80-100%): mean={np.mean(late_gripper):.3f}, std={np.std(late_gripper):.3f}")

# Check first few steps of first demo
first_demo = f['data'][demos[0]]['actions'][:]
print(f"\nFirst demo - first 20 gripper actions:")
print(first_demo[:20, 6])

f.close()