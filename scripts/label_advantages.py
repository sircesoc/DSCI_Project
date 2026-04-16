#!/usr/bin/env python
"""Compute per-frame advantage weights from keypoint progress.

For each frame, measures how much the garment keypoints moved toward the
folded configuration compared to the previous frame. Frames where progress
is made get high weights; stagnant or regressive frames get low weights.

The advantage is computed as the negative change in a "fold distance" metric
— the sum of relevant keypoint pair distances that should decrease during folding.

Garment-specific fold distance:
  top-long/top-short:  dist(0,4) + dist(1,5) + dist(2,3)
  long-pant:           dist(0,4) + dist(1,5) + dist(2,3)
  short-pant:          dist(0,4) + dist(1,5) + dist(2,3)

All garment types use the same keypoint pairs since the fold targets are
symmetric (left-right fold + body fold).

Output: adds `advantage_weight` column (float32, shape [1]) to the dataset.

Usage:
    python scripts/label_advantages.py \
        --src_root /path/to/four_types_merged
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def compute_advantage_weights(df: pd.DataFrame, window: int = 16) -> np.ndarray:
    """Compute per-frame advantage weights for all episodes in a dataframe.

    Uses a sliding window to measure keypoint progress over `window` frames,
    matching the action chunk size for meaningful advantage signals.

    Returns:
        weights: (N,) float32 array of advantage weights in [0, 1].
    """
    weights = np.full(len(df), 0.5, dtype=np.float32)  # default: neutral

    for ep_idx in df["episode_index"].unique():
        mask = df["episode_index"] == ep_idx
        ep = df[mask]
        indices = np.where(mask)[0]

        # Stack keypoints: (T, 6, 2)
        kp_list = ep["observation.keypoints"].tolist()
        kp = np.array([np.stack(k) for k in kp_list])  # (T, 6, 2)
        T = len(kp)

        # Fold distance: sum of keypoint pair distances that should decrease
        d04 = np.linalg.norm(kp[:, 0] - kp[:, 4], axis=-1)  # (T,)
        d15 = np.linalg.norm(kp[:, 1] - kp[:, 5], axis=-1)
        d23 = np.linalg.norm(kp[:, 2] - kp[:, 3], axis=-1)
        fold_dist = d04 + d15 + d23  # (T,)

        # Sliding window advantage: progress over `window` frames
        delta = np.zeros(T, dtype=np.float32)
        for i in range(T):
            j = min(i + window, T - 1)
            delta[i] = -(fold_dist[j] - fold_dist[i])  # positive when distance decreases

        # Normalize to [0, 1] using percentile-based scaling
        if delta.std() > 1e-8:
            # Clip outliers at 5th/95th percentile, then scale to [0.1, 1.0]
            lo = np.percentile(delta, 10)
            hi = np.percentile(delta, 90)
            if hi > lo:
                scaled = np.clip((delta - lo) / (hi - lo), 0.0, 1.0)
                frame_weights = 0.1 + 0.9 * scaled  # map to [0.1, 1.0]
            else:
                frame_weights = np.full(T, 0.5, dtype=np.float32)
        else:
            frame_weights = np.full(T, 0.5, dtype=np.float32)

        weights[indices] = frame_weights

    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True,
                        help="Path to the dataset root")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    data_dir = src_root / "data"

    # Process all parquet files
    for pq_path in sorted(data_dir.rglob("*.parquet")):
        print(f"Processing {pq_path}")
        df = pd.read_parquet(pq_path)

        if "advantage_weight" in df.columns:
            print("  advantage_weight exists, recomputing")
            df = df.drop(columns=["advantage_weight"])

        weights = compute_advantage_weights(df)

        # Print distribution
        prog = (weights > 0.8).sum()
        stag = ((weights > 0.3) & (weights < 0.8)).sum()
        regr = (weights < 0.3).sum()
        print(f"  Progressive: {prog/len(weights)*100:.1f}%, "
              f"Stagnant: {stag/len(weights)*100:.1f}%, "
              f"Regressive: {regr/len(weights)*100:.1f}%")

        # Add as list column (shape [1]) to match LeRobot convention
        df["advantage_weight"] = [[w] for w in weights]

        # Write back
        table = pa.Table.from_pandas(df)
        pq.write_table(table, pq_path)
        print(f"  Saved with advantage_weight column")

    # Patch meta/info.json
    info_path = src_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if "observation.advantage_weight" not in info.get("features", {}):
        info["features"]["observation.advantage_weight"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Patched {info_path}")

    print("Done!")


if __name__ == "__main__":
    main()
