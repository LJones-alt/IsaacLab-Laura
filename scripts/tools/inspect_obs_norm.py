import h5py
import numpy as np

file_path = "/workspace/isaaclab/docs/place_VM/beaker_place_sm_demos_generated.hdf5"

try:
    with h5py.File(file_path, "r") as f:
        if "obs_normalization" in f:
            print("Found obs_normalization group!")
            for obs_key in f["obs_normalization"].keys():
                print(f"\n--- {obs_key} ---")
                stats = f["obs_normalization"][obs_key]
                for stat_name in stats.keys():
                    val = stats[stat_name][:]
                    print(f"  {stat_name}: {list(np.round(val, 4))}")
        else:
            print("No 'obs_normalization' group found in HDF5.")
except Exception as e:
    print(f"Error reading {file_path}: {e}")
