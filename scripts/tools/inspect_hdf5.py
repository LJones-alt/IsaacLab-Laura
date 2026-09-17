import h5py
import numpy as np

file_path = "/workspace/isaaclab/docs/place_VM/beaker_place_sm_demos2.hdf5"

try:
    with h5py.File(file_path, "r") as f:
        print("Data groups in file:")
        print(list(f.keys()))
        if "data" in f:
            demos = list(f["data"].keys())
            print(f"Total demos: {len(demos)}")
            
            # Analyze actions across a few demos
            all_actions = []
            for i, demo in enumerate(demos[:5]):  # check first 5 demos
                if "actions" in f["data"][demo]:
                    actions = f["data"][demo]["actions"][:]
                    all_actions.append(actions)
                    sys_actions = actions
                    print(f"\n{demo} actions shape: {actions.shape}")
                    print(f"Sample actions (first 3 steps):")
                    for step in range(min(3, len(actions))):
                        print(np.round(actions[step], 4))
            
            if all_actions:
                all_actions = np.vstack(all_actions)
                print(f"\nAcross first 5 demos (total steps: {len(all_actions)}):")
                print(f"Min values per dimension: {np.round(np.min(all_actions, axis=0), 4)}")
                print(f"Max values per dimension: {np.round(np.max(all_actions, axis=0), 4)}")
                print(f"Mean values per dimension: {np.round(np.mean(all_actions, axis=0), 4)}")
                print(f"Std values per dimension: {np.round(np.std(all_actions, axis=0), 4)}")
                
        else:
            print("No 'data' group found in HDF5.")
except Exception as e:
    print(f"Error reading {file_path}: {e}")
