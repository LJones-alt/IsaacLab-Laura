import h5py
import numpy as np

with h5py.File("docs/place_VM/beaker_place_sm_demos2.hdf5", "r") as f:
    demo_key = list(f["data"].keys())[0]
    joint_pos_demo = f[f"data/{demo_key}/obs/joint_pos"][:]
    print(f"HDF5 'joint_pos' - Min: {np.min(joint_pos_demo):.4f}, Max: {np.max(joint_pos_demo):.4f}")