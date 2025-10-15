import h5py
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "docs/beaker_lift_new_start_skillgen_annotated.hdf5"
OUTPUT_FILE = "docs/subset_first5_clean.hdf5"
DATA_GROUP = "data"
DEMO_PREFIX = "demo_"
NUM_DEMOS = 5

# Metadata groups to copy (common in IsaacLab)
METADATA_GROUPS = ["meta", "env", "info", "config", "episodes", "stats"]

# -----------------------------
# STEP 1: Create subset HDF5
# -----------------------------
with h5py.File(INPUT_FILE, "r") as src, h5py.File(OUTPUT_FILE, "w") as dst:
    # 1. Copy root attributes
    for k, v in src.attrs.items():
        dst.attrs[k] = v

    # 2. Copy metadata groups
    for g in METADATA_GROUPS:
        if g in src:
            src.copy(src[g], dst, name=g)
            print(f"✅ Copied metadata group: /{g}")

    # 3. Copy /data group and first N demos
    src_data = src[DATA_GROUP]
    dst_data = dst.create_group(DATA_GROUP)

    demo_keys = sorted([k for k in src_data.keys() if k.startswith(DEMO_PREFIX)])[:NUM_DEMOS]

    for key in demo_keys:
        src.copy(src_data[key], dst_data, name=key)
        print(f"✅ Copied demo: /{DATA_GROUP}/{key}")

    # -----------------------------
    # STEP 2: Trim metadata arrays to match only the first N demos
    # -----------------------------
    # Example metadata: episodes/episode_lengths, episodes/episode_starts, etc.
    if "episodes" in dst:
        episodes_group = dst["episodes"]
        for name, dataset in episodes_group.items():
            if isinstance(dataset, h5py.Dataset):
                # Trim first NUM_DEMOS entries
                new_data = dataset[0:NUM_DEMOS]
                del episodes_group[name]
                episodes_group.create_dataset(name, data=new_data)
                print(f"✅ Trimmed episodes/{name} to first {NUM_DEMOS} entries")

    # Some attributes may reference total episodes/envs
    for attr_key in ["num_episodes", "num_envs"]:
        if attr_key in dst.attrs:
            dst.attrs[attr_key] = NUM_DEMOS
            print(f"✅ Updated root attribute: {attr_key}={NUM_DEMOS}")

print(f"\n🎉 Subset with {NUM_DEMOS} demos created successfully: {OUTPUT_FILE}")
