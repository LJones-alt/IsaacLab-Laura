from __future__ import annotations

"""
Normalize low-dimensional observations in a robomimic HDF5 dataset.

Statistics are computed ONLY from the train split, then applied to all demos
(train + valid) to avoid data leakage. Normalization stats are saved inside
the HDF5 file so they can be loaded at inference time for denormalization.

Usage:
    python docs/normalize_low_dim_obs.py \
        --dataset docs/place_VM/beaker_place_sm_demos_generated.hdf5

The script modifies the file in-place. A backup is written alongside it
as <dataset>_backup.hdf5 before any changes are made.
"""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


# ── Keys to normalize ────────────────────────────────────────────────────────
# Edit this list if your dataset has different low_dim keys.
LOW_DIM_KEYS = [
    "eef_pos",
    "eef_quat",
    "gripper_pos",
    "object_position",
    "target_object_position",
]

# Minimum std value to avoid division by near-zero
EPS = 1e-8


def collect_stats(f: h5py.File, demo_keys: list[str], obs_keys: list[str]) -> dict:
    """Collect per-key mean and std from the given demonstrations."""
    accum: dict[str, list[np.ndarray]] = {k: [] for k in obs_keys}

    for demo in demo_keys:
        obs_grp = f[f"data/{demo}/obs"]
        for key in obs_keys:
            if key in obs_grp:
                accum[key].append(obs_grp[key][:])  # (T, dim)

    stats = {}
    for key, arrays in accum.items():
        if not arrays:
            print(f"  [WARN] key '{key}' not found in any demo, skipping.")
            continue
        data = np.concatenate(arrays, axis=0)  # (N_total, dim)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < EPS, 1.0, std)  # don't divide by ~0
        stats[key] = {"mean": mean, "std": std}
        print(f"  {key:30s} mean={mean.round(4)}  std={std.round(4)}")

    return stats


def normalize_demos(f: h5py.File, demo_keys: list[str], stats: dict) -> None:
    """Normalize obs in-place for the given demos using precomputed stats."""
    for demo in demo_keys:
        obs_grp = f[f"data/{demo}/obs"]
        for key, s in stats.items():
            if key not in obs_grp:
                continue
            data = obs_grp[key][:]
            normalized = (data - s["mean"]) / s["std"]
            obs_grp[key][...] = normalized.astype(data.dtype)


def save_stats(f: h5py.File, stats: dict) -> None:
    """Store normalization stats in the HDF5 file for later use."""
    grp = f.require_group("obs_normalization")
    for key, s in stats.items():
        key_grp = grp.require_group(key)
        for stat_name, value in s.items():
            if stat_name in key_grp:
                del key_grp[stat_name]
            key_grp.create_dataset(stat_name, data=value)
    print("\nNormalization stats saved to 'obs_normalization' group in HDF5.")


def get_demo_keys(f: h5py.File, filter_key: str | None) -> list[str]:
    """Return sorted demo keys, optionally filtered by a mask dataset."""
    if filter_key and filter_key in f["mask"]:
        # robomimic stores demo indices as bytes, e.g. b"demo_0"
        raw = f[f"mask/{filter_key}"][:]
        keys = [r.decode("utf-8") if isinstance(r, bytes) else r for r in raw]
        return sorted(keys)
    else:
        return sorted(f["data"].keys())


def main():
    parser = argparse.ArgumentParser(description="Normalize low-dim obs in HDF5 dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the HDF5 file.")
    parser.add_argument(
        "--train-filter-key",
        default="train",
        help="HDF5 mask key for train split (default: 'train').",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a backup file (not recommended).",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=LOW_DIM_KEYS,
        help="Low-dim obs keys to normalize.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # ── Backup ────────────────────────────────────────────────────────────────
    if not args.no_backup:
        backup_path = dataset_path.parent / (dataset_path.stem + "_backup.hdf5")
        if not backup_path.exists():
            print(f"Writing backup to {backup_path} ...")
            shutil.copy2(dataset_path, backup_path)
        else:
            print(f"Backup already exists at {backup_path}, skipping.")

    with h5py.File(dataset_path, "r+") as f:
        # ── Get demo splits ───────────────────────────────────────────────────
        train_demos = get_demo_keys(f, args.train_filter_key)
        all_demos = sorted(f["data"].keys())
        n_train = len(train_demos)
        n_all = len(all_demos)
        print(f"\nDataset: {dataset_path.name}")
        print(f"  Total demos : {n_all}")
        print(f"  Train demos : {n_train} (stats computed from these only)")
        print(f"  Other demos : {n_all - n_train} (normalized using train stats)\n")

        # ── Compute stats from train split only ───────────────────────────────
        print("Computing normalization statistics from train split:")
        stats = collect_stats(f, train_demos, args.keys)

        # ── Normalize all demos ───────────────────────────────────────────────
        print("\nNormalizing train demos ...")
        normalize_demos(f, train_demos, stats)

        other_demos = [d for d in all_demos if d not in train_demos]
        if other_demos:
            print(f"Normalizing {len(other_demos)} non-train demos ...")
            normalize_demos(f, other_demos, stats)

        # ── Save stats ────────────────────────────────────────────────────────
        save_stats(f, stats)

    print("\nDone. To denormalize predictions at inference time:")
    print("  obs_pred = obs_normalized * std + mean")
    print("  (stats are stored in the HDF5 under 'obs_normalization/<key>/mean|std')")


if __name__ == "__main__":
    main()
