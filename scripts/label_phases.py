"""Add observation.phase column to four_types_merged dataset.

Detects fold phases (0/1/2) from gripper close→open cycles in the action
column. Each completed gripper cycle advances the phase.

No simulation required — pure parquet manipulation.

Usage
-----
    python scripts/label_phases.py \
        --src_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Gripper cycle detector (same logic as recovery_lerobot_policy.py)
# ---------------------------------------------------------------------------

class GripperCycleDetector:
    """Detects a full grasp-release cycle on a single gripper.

    A cycle is: value drops below close_thresh (grasp) then rises above
    open_thresh (release).

    Thresholds calibrated for raw action values (not normalized state):
    - Closed: ~-0.15
    - Open:   ~0.3-0.6
    """

    def __init__(self, close_thresh: float = -0.05, open_thresh: float = 0.25,
                 min_close_steps: int = 3):
        self.close_thresh = close_thresh
        self.open_thresh = open_thresh
        self.min_close_steps = min_close_steps
        self._was_closed = False
        self._close_steps = 0

    def reset(self):
        self._was_closed = False
        self._close_steps = 0

    def update(self, value: float) -> bool:
        """Feed one gripper reading. Returns True on completed cycle."""
        if value < self.close_thresh:
            self._close_steps += 1
            if self._close_steps >= self.min_close_steps:
                self._was_closed = True
        else:
            self._close_steps = 0

        if self._was_closed and value > self.open_thresh:
            self._was_closed = False
            self._close_steps = 0
            return True  # cycle complete

        return False


LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11
MAX_PHASE = 2


def label_episode_phases(actions: np.ndarray) -> np.ndarray:
    """Label each frame with a phase (0/1/2) based on gripper cycles.

    Args:
        actions: (T, action_dim) float32 array.

    Returns:
        (T,) int32 array of phase labels.
    """
    T = len(actions)
    phases = np.zeros(T, dtype=np.int32)
    left_det = GripperCycleDetector()
    right_det = GripperCycleDetector()
    phase = 0

    for t in range(T):
        left_cycle = left_det.update(float(actions[t, LEFT_GRIPPER_IDX]))
        right_cycle = right_det.update(float(actions[t, RIGHT_GRIPPER_IDX]))
        if left_cycle or right_cycle:
            phase = min(phase + 1, MAX_PHASE)
        phases[t] = phase

    return phases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add observation.phase to dataset.")
    p.add_argument("--src_root", required=True,
                   help="Dataset root (four_types_merged). Modified in-place.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src_root)

    # Load parquet
    data_dir = src / "data" / "chunk-000"
    parquet_file = sorted(data_dir.glob("*.parquet"))[0]
    print(f"[LabelPhase] Reading parquet: {parquet_file}")
    table = pq.read_table(str(parquet_file))
    print(f"[LabelPhase] Rows: {len(table)}  Columns: {table.column_names}")

    if "observation.phase" in table.column_names:
        print("[LabelPhase] observation.phase already exists — removing to re-label.")
        idx = table.column_names.index("observation.phase")
        table = table.remove_column(idx)

    # Group by episode
    ep_col = table["episode_index"].to_pylist()
    act_col = table["action"].to_pylist()

    episodes: dict[int, list[int]] = {}
    for i, ep in enumerate(ep_col):
        episodes.setdefault(ep, []).append(i)

    # Label phases per episode
    all_phases = [0] * len(table)
    phase_stats: dict[int, Counter] = {}  # ep -> Counter of phase values

    for ep_idx in sorted(episodes.keys()):
        row_indices = episodes[ep_idx]
        actions = np.array([act_col[i] for i in row_indices], dtype=np.float32)
        phases = label_episode_phases(actions)

        for local_t, global_i in enumerate(row_indices):
            all_phases[global_i] = int(phases[local_t])

        phase_stats[ep_idx] = Counter(phases.tolist())

    # Print summary
    total_by_phase = Counter()
    stuck_at_0 = 0
    for ep_idx, counts in sorted(phase_stats.items()):
        total_by_phase.update(counts)
        if max(counts.keys()) == 0:
            stuck_at_0 += 1

    n_eps = len(episodes)
    print(f"\n[LabelPhase] Phase distribution across {n_eps} episodes:")
    for ph in sorted(total_by_phase.keys()):
        pct = 100.0 * total_by_phase[ph] / len(table)
        print(f"  Phase {ph}: {total_by_phase[ph]} frames ({pct:.1f}%)")
    print(f"  Episodes stuck at phase 0: {stuck_at_0}/{n_eps}")

    # Per garment type breakdown (250 eps each: 0-249, 250-499, 500-749, 750-999)
    type_names = ["top-short (0-249)", "top-long (250-499)",
                  "short-pant (500-749)", "long-pant (750-999)"]
    for i, name in enumerate(type_names):
        start, end = i * 250, (i + 1) * 250
        type_counter = Counter()
        for ep_idx in range(start, min(end, n_eps)):
            if ep_idx in phase_stats:
                type_counter.update(phase_stats[ep_idx])
        total_frames = sum(type_counter.values())
        if total_frames > 0:
            parts = [f"ph{p}={type_counter.get(p,0)} ({100*type_counter.get(p,0)/total_frames:.0f}%)"
                     for p in range(3)]
            print(f"  {name}: {', '.join(parts)}")

    # Add column to parquet
    phase_col = pa.array(
        [[v] for v in all_phases],
        type=pa.list_(pa.int32()),
    )
    new_table = table.append_column(
        pa.field("observation.phase", pa.list_(pa.int32())),
        phase_col,
    )

    # Write in-place
    pq.write_table(new_table, str(parquet_file))
    print(f"\n[LabelPhase] Wrote parquet in-place: {parquet_file}")

    # Patch info.json
    info_path = src / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info.setdefault("features", {})["observation.phase"] = {
            "dtype": "int32",
            "shape": [1],
            "names": ["phase_idx"],
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print("[LabelPhase] Patched info.json")

    print(f"\n[LabelPhase] Done. Column observation.phase added (shape=[1], values 0-2).")


if __name__ == "__main__":
    main()
