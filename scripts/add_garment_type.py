"""Add observation.garment_type column to four_types_merged dataset.

Reads garment_info.json to determine each episode's garment type (0-3),
then writes a new dataset with that column added.  No simulation required.

Output: four_types_merged_gt  (gt = garment-type)

Usage
-----
    python scripts/add_garment_type.py \\
        --src_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged \\
        --dst_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged_gt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


GARMENT_TYPE_TO_IDX = {
    "top-long-sleeve":  0,
    "top-short-sleeve": 1,
    "long-pant":        2,
    "short-pant":       3,
}

_NAME_PREFIX_TO_TYPE = {
    "Top_Long":  "top-long-sleeve",
    "Top_Short": "top-short-sleeve",
    "Pant_Long": "long-pant",
    "Pant_Short": "short-pant",
}


def garment_name_to_type(garment_name: str) -> str:
    parts = garment_name.split("_")
    prefix = "_".join(parts[:2])
    return _NAME_PREFIX_TO_TYPE.get(prefix, "top-long-sleeve")


def build_episode_type_map(garment_info: dict) -> dict[int, int]:
    """Build {global_episode_idx: type_idx} from garment_info.json."""
    ep_map: dict[int, int] = {}
    ep_idx = 0
    for garment_name, episodes in garment_info.items():
        type_str = garment_name_to_type(garment_name)
        type_idx = GARMENT_TYPE_TO_IDX.get(type_str, 0)
        for _ in range(len(episodes)):
            ep_map[ep_idx] = type_idx
            ep_idx += 1
    return ep_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True)
    p.add_argument("--dst_root", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src_root)
    dst = Path(args.dst_root)
    dst.mkdir(parents=True, exist_ok=True)

    # Load garment_info
    with open(src / "meta" / "garment_info.json") as f:
        garment_info = json.load(f)

    ep_type_map = build_episode_type_map(garment_info)
    print(f"[AddGT] Episode→type_idx map built for {len(ep_type_map)} episodes")
    # Sanity check
    from collections import Counter
    counts = Counter(ep_type_map.values())
    print(f"[AddGT] Type distribution: {dict(sorted(counts.items()))}")

    # ----- Parquet -----
    data_src = src / "data" / "chunk-000"
    data_dst = dst / "data" / "chunk-000"
    data_dst.mkdir(parents=True, exist_ok=True)

    parquet_file = sorted(data_src.glob("*.parquet"))[0]
    print(f"[AddGT] Reading parquet: {parquet_file}")
    table = pq.read_table(str(parquet_file))
    print(f"[AddGT] Rows: {len(table)}  Columns: {table.column_names}")

    # Build garment_type column (one int per row, based on episode_index)
    ep_indices = table["episode_index"].to_pylist()
    garment_type_vals = [ep_type_map.get(ep, 0) for ep in ep_indices]

    # Add as list-type column (shape [1] per row, matching LeRobot STATE convention)
    garment_type_col = pa.array(
        [[v] for v in garment_type_vals],
        type=pa.list_(pa.int32()),
    )
    new_table = table.append_column(
        pa.field("observation.garment_type", pa.list_(pa.int32())),
        garment_type_col,
    )

    out_parquet = data_dst / parquet_file.name
    pq.write_table(new_table, str(out_parquet))
    print(f"[AddGT] Wrote parquet: {out_parquet}")

    # ----- Videos — symlink to avoid copying ~100 GB -----
    videos_src = src / "videos"
    videos_dst = dst / "videos"
    if videos_src.exists():
        if videos_dst.exists() or videos_dst.is_symlink():
            videos_dst.unlink() if videos_dst.is_symlink() else shutil.rmtree(videos_dst)
        os.symlink(videos_src.resolve(), videos_dst)
        print(f"[AddGT] Symlinked videos: {videos_dst} → {videos_src.resolve()}")

    # ----- Meta -----
    meta_dst = dst / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    for f in (src / "meta").iterdir():
        shutil.copy2(f, meta_dst / f.name)

    # Patch info.json to add the new feature
    info_path = meta_dst / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info.setdefault("features", {})["observation.garment_type"] = {
            "dtype": "int32",
            "shape": [1],
            "names": ["type_idx"],
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[AddGT] Patched info.json")

    print(f"\n[AddGT] Done. Dataset written to {dst}")
    print(f"[AddGT] New column: observation.garment_type  shape=[1]  values 0-3")


if __name__ == "__main__":
    main()
