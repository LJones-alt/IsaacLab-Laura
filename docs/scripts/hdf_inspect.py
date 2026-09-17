import h5py
import numpy as np

dataset_path = "docs/place_VM/place_vial_demos_generated.hdf5"

with h5py.File(dataset_path, "r") as f:
    demos = sorted(f["data"].keys())

    all_actions = []

    for demo in demos:
        actions = np.asarray(f[f"data/{demo}/actions"])
        all_actions.append(actions)

    all_actions = np.concatenate(all_actions, axis=0)

    print(f"Number of demos: {len(demos)}")
    print(f"Total action samples: {len(all_actions)}")
    print(f"Action shape: {all_actions.shape}")
    print()

    print("GLOBAL ACTION RANGE")
    print(f"min = {all_actions.min()}")
    print(f"max = {all_actions.max()}")
    print()

    print("PER-DIMENSION ACTION RANGE")
    for i in range(all_actions.shape[1]):
        print(
            f"action[{i}]: "
            f"min = {all_actions[:, i].min(): .6f}, "
            f"max = {all_actions[:, i].max(): .6f}"
        )