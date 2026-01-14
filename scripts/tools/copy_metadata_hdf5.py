import h5py
import json

# File and group
HDF5_FILE = "docs/subset_first5.hdf5"
DATA_GROUP = "data"

# Metadata to add
# robomimic v0.5 requires 'env_kwargs' key (not 'sim_args')
metadata = {
    "env_name": "Cube-Mimic-v0",
    "type": 2,
    "env_kwargs": {
        "dt": 0.01,
        "decimation": 2,
        "render_interval": 2,
        "num_envs": 1
    }
}

# Open file in append mode
with h5py.File(HDF5_FILE, "a") as f:
    data_group = f[DATA_GROUP]

    # Store each top-level key as an attribute
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Convert nested dicts to JSON string
            data_group.attrs[key] = json.dumps(value)
        else:
            data_group.attrs[key] = value

print(f"✅ Metadata added to /{DATA_GROUP}")
