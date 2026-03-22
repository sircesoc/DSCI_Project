"""Merge per-episode keypoint labels into an existing LeRobotDataset in-place.

Reads the kp_labels directory produced by label_keypoints.py, and adds the
observation.keypoints column directly to the source parquet file.

Does NOT create a separate dataset or symlink videos (avoids exFAT issues).
Assumes observation.garment_type already exists in the dataset.

Usage
-----
    python scripts/merge_keypoint_labels.py \\
        --dataset_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged \\
        --kp_labels_dir /media/sircesoc/WD_BLACK/lehome/kp_labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True,
                   help="Dataset root to patch in-place.")
    p.add_argument("--kp_labels_dir", required=True,
                   help="Directory with per-episode .npy keypoint files from label_keypoints.py.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    kp_dir = Path(args.kp_labels_dir)

    # ------------------------------------------------------------------
    # Load kp meta
    # ------------------------------------------------------------------
    with open(kp_dir / "meta.json") as f:
        kp_meta = {int(k): v for k, v in json.load(f).items()}
    print(f"[MergeKP] Loaded meta for {len(kp_meta)} episodes")

    # ------------------------------------------------------------------
    # Read source parquet
    # ------------------------------------------------------------------
    parquet_path = sorted((root / "data" / "chunk-000").glob("*.parquet"))[0]
    print(f"[MergeKP] Reading parquet: {parquet_path}")
    table = pq.read_table(str(parquet_path))
    total_frames = len(table)
    print(f"[MergeKP] {total_frames} frames")

    # Check if keypoints column already exists
    if "observation.keypoints" in table.column_names:
        print("[MergeKP] observation.keypoints already exists — dropping old column")
        table = table.drop(["observation.keypoints"])

    # ------------------------------------------------------------------
    # Build per-frame keypoint array
    # ------------------------------------------------------------------
    ep_col = table["episode_index"].to_pylist()
    idx_col = table["index"].to_pylist()

    # Build episode → frame row mapping
    ep_to_rows: dict[int, list[int]] = {}
    for row_i, ep_idx in enumerate(ep_col):
        ep_to_rows.setdefault(ep_idx, []).append(row_i)

    kp_array = np.full((total_frames, 6, 2), -1.0, dtype=np.float32)
    labeled_eps = 0

    for ep_idx, meta in kp_meta.items():
        kp_file = kp_dir / f"episode_{ep_idx:06d}.npy"
        if not kp_file.exists():
            continue

        ep_kp = np.load(str(kp_file))  # (T, 6, 2)
        rows = ep_to_rows.get(ep_idx, [])
        T = min(len(ep_kp), len(rows))

        for t in range(T):
            kp_array[rows[t]] = ep_kp[t]

        labeled_eps += 1

    print(f"[MergeKP] Labeled {labeled_eps}/{len(kp_meta)} episodes")

    # ------------------------------------------------------------------
    # Add keypoints column to parquet
    # ------------------------------------------------------------------
    # Store as fixed-size list<list<float32>> matching shape (6, 2)
    kp_list = [[row.tolist() for row in kp_array[i]] for i in range(total_frames)]
    kp_col = pa.array(kp_list, type=pa.list_(pa.list_(pa.float32(), 2), 6))
    new_table = table.append_column("observation.keypoints", kp_col)

    # Write back in-place
    print(f"[MergeKP] Writing patched parquet back to {parquet_path} ...")
    pq.write_table(new_table, str(parquet_path))

    # ------------------------------------------------------------------
    # Update meta/info.json
    # ------------------------------------------------------------------
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info.setdefault("features", {})["observation.keypoints"] = {
        "dtype": "float32",
        "shape": [6, 2],
        "names": ["keypoint", "uv"],
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print("[MergeKP] Patched info.json")

    # ------------------------------------------------------------------
    # Update meta/stats.json
    # ------------------------------------------------------------------
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        # Compute actual stats from labeled keypoints (excluding -1 invalid)
        valid_mask = (kp_array != -1.0).all(axis=-1)  # (N, 6)
        valid_kp = kp_array[valid_mask.any(axis=-1)]  # frames with any valid kp
        if len(valid_kp) > 0:
            # Stats per coordinate (u, v)
            valid_flat = kp_array[np.broadcast_to(valid_mask[..., None], kp_array.shape)]
            valid_flat = valid_flat.reshape(-1, 2)
            stats["observation.keypoints"] = {
                "min": valid_flat.min(axis=0).tolist(),
                "max": valid_flat.max(axis=0).tolist(),
                "mean": valid_flat.mean(axis=0).tolist(),
                "std": valid_flat.std(axis=0).tolist(),
            }
        else:
            stats["observation.keypoints"] = {
                "min": [0.0, 0.0], "max": [1.0, 1.0],
                "mean": [0.5, 0.5], "std": [0.3, 0.3],
            }

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print("[MergeKP] Patched stats.json")

    print(f"\n[MergeKP] Done. Keypoints added in-place to {root}")


if __name__ == "__main__":
    main()
