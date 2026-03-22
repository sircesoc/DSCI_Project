"""Retroactively label keypoints for existing four_types_merged episodes.

Replays each episode in simulation, reads garment mesh check_points at each frame,
projects to 2D, and saves per-episode keypoint arrays + garment type labels.

Output layout
-------------
    <output_dir>/
        episode_000000.npy    # (T, 6, 2) float32  — normalized uv, -1 = invalid
        episode_000001.npy
        ...
        meta.json             # {episode_idx: {garment_name, garment_type_str, type_idx}}

After this script completes, run merge_keypoint_labels.py to produce the
four_types_merged_kp LeRobotDataset.

Usage (launch via AppLauncher like other Isaac scripts)
-------------------------------------------------------
    python -m scripts.label_keypoints \\
        --task LeHome-BiSO101-Direct-Garment-v2 \\
        --src_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged \\
        --output_dir /media/sircesoc/WD_BLACK/lehome/kp_labels \\
        --garment_cfg_base_path /path/to/garment_cfgs \\
        --particle_cfg_path /path/to/particle_cfg \\
        [--start_episode 0] [--end_episode 1000] \\
        --headless
"""

from __future__ import annotations

import multiprocessing

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retroactively label garment keypoints.")
    p.add_argument("--task", default="LeHome-BiSO101-Direct-Garment-v2")
    p.add_argument("--src_root", required=True,
                   help="Source dataset root (four_types_merged).")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write per-episode .npy keypoint files.")
    p.add_argument("--garment_cfg_base_path", default="Assets/objects/Challenge_Garment")
    p.add_argument("--particle_cfg_path",
                   default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml")
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--end_episode", type=int, default=None)
    # Note: AppLauncher.add_app_launcher_args(p) adds --device (default "cpu").
    # We reuse that for the env_cfg device; do NOT add a separate --device arg.
    p.add_argument("--stabilize_steps", type=int, default=30,
                   help="Physics steps after reset before replaying actions.")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Garment type helpers
# ---------------------------------------------------------------------------

GARMENT_TYPE_TO_IDX = {
    "top-long-sleeve":  0,
    "top-short-sleeve": 1,
    "long-pant":        2,
    "short-pant":       3,
}

_GARMENT_NAME_PREFIX_TO_TYPE = {
    "Top_Long":  "top-long-sleeve",
    "Top_Short": "top-short-sleeve",
    "Pant_Long": "long-pant",
    "Pant_Short": "short-pant",
}


def garment_name_to_type_str(garment_name: str) -> str:
    """Infer garment type string from garment name.

    e.g. "Top_Short_Seen_3" → "top-short-sleeve"
    """
    parts = garment_name.split("_")
    prefix = "_".join(parts[:2])  # "Top_Short", "Pant_Long", etc.
    return _GARMENT_NAME_PREFIX_TO_TYPE.get(prefix, "top-long-sleeve")


def garment_name_to_version(garment_name: str) -> str:  # noqa: ARG001
    """Return the filesystem version directory for a garment.

    All challenge garments live under 'Release' regardless of their
    Seen/Unseen split label in the name.
    """
    return "Release"


# ---------------------------------------------------------------------------
# 3-D → 2-D projection (top camera, matches dataset_record.py)
# ---------------------------------------------------------------------------

import numpy as np
from scipy.spatial.transform import Rotation as _ScipyRotation

_R_w2r = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
_t_w2r = np.array([0.23, -0.25, 0.5], dtype=np.float32)

_R_usd = _ScipyRotation.from_quat([-0.9862856, 0, 0, 0.1650476]).as_matrix().astype(np.float32)
_R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
_R_mix = _R_usd @ _R_opt
_t_cam = np.array([0.225, -0.5, 0.6], dtype=np.float32)

_FX = _FY = 482.0
_CX, _CY = 320.0, 240.0
_IMG_W, _IMG_H = 640, 480


def project_kp_3d_to_2d(kp_3d_cm: np.ndarray) -> np.ndarray:
    """Project garment check_points (cm, world frame) to normalized image coords.

    Args:
        kp_3d_cm: (N, 3) keypoints in centimetres, world frame.

    Returns:
        (N, 2) float32 in [0,1]; points behind camera or out-of-frame → -1.
    """
    kp_3d_m = np.array(kp_3d_cm, dtype=np.float32) / 100.0

    # World → RobotBase
    kp_robot = (kp_3d_m - _t_w2r) @ _R_w2r.T  # (N, 3)

    # RobotBase → Camera
    kp_cam = (kp_robot - _t_cam) @ _R_mix  # (N, 3)

    # Camera convention: -z is forward (matches dataset_record.py).
    # Project directly without z > 0 check.
    z = kp_cam[:, 2]
    nonzero = np.abs(z) > 1e-6
    kp_2d = np.full((len(kp_3d_m), 2), -1.0, dtype=np.float32)
    if nonzero.any():
        u = (_FX * kp_cam[nonzero, 0] / z[nonzero] + _CX) / _IMG_W
        v = (_FY * kp_cam[nonzero, 1] / z[nonzero] + _CY) / _IMG_H
        kp_2d[nonzero, 0] = u
        kp_2d[nonzero, 1] = v

    # Mark out-of-frame as invalid
    out_of_frame = (kp_2d[:, 0] < 0) | (kp_2d[:, 0] > 1) | \
                   (kp_2d[:, 1] < 0) | (kp_2d[:, 1] > 1)
    kp_2d[out_of_frame] = -1.0

    return kp_2d


# ---------------------------------------------------------------------------
# Main labelling logic
# ---------------------------------------------------------------------------

def build_episode_garment_map(garment_info: dict) -> list[tuple[str, int]]:
    """Build a flat list mapping global episode index → (garment_name, sub_ep_idx).

    garment_info structure:
        { garment_name: { "0": {...}, "1": {...}, ..., "24": {...} } }

    Episodes in the dataset are ordered: all episodes of garment_0, then garment_1, etc.
    """
    ep_map: list[tuple[str, int]] = []
    for garment_name, episodes in garment_info.items():
        n = len(episodes)
        for sub_idx in range(n):
            ep_map.append((garment_name, sub_idx))
    return ep_map


def preload_all_actions(parquet_path: str) -> dict[int, np.ndarray]:
    """Load ALL episodes' actions from parquet in a single scan.

    Returns a dict {episode_idx: (T, action_dim) float32 array}.
    """
    import pyarrow.parquet as pq

    print(f"[LabelKP] Preloading all actions from parquet (single scan)...")
    table = pq.read_table(parquet_path, columns=["episode_index", "action"])
    ep_col = table["episode_index"].to_pylist()
    act_col = table["action"].to_pylist()

    all_actions: dict[int, list] = {}
    for ep_idx, action in zip(ep_col, act_col):
        all_actions.setdefault(ep_idx, []).append(action)

    result = {ep: np.array(acts, dtype=np.float32) for ep, acts in all_actions.items()}
    print(f"[LabelKP] Loaded actions for {len(result)} episodes.")
    return result


def _physics_step(env, action_tensor=None, render=True) -> None:
    """Step physics. Use render=True when reading cloth positions afterward.

    render=True syncs USD cloth attributes so _get_points_pose() returns
    current values. render=False is faster but gives stale cloth positions.
    """
    try:
        if action_tensor is not None:
            env._pre_physics_step(action_tensor)
        substeps = getattr(getattr(env, "cfg", None), "decimation", 1)
        for _ in range(substeps):
            env.sim.step(render=render)
    except Exception:
        import torch
        act = action_tensor if action_tensor is not None else torch.zeros(1, 12, device=env.device)
        env.step(act)


def label_all_episodes(args, simulation_app) -> None:
    import json
    import time
    import torch
    import gymnasium as gym
    from pathlib import Path
    from isaaclab_tasks.utils import parse_env_cfg
    from lehome.utils.success_checker_chanllege import get_object_particle_position

    src_root = Path(args.src_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load garment_info
    with open(src_root / "meta" / "garment_info.json") as f:
        garment_info = json.load(f)

    ep_map = build_episode_garment_map(garment_info)
    n_total = len(ep_map)
    end = args.end_episode if args.end_episode is not None else n_total
    print(f"[LabelKP] Total episodes: {n_total}, labelling {args.start_episode}–{end-1}")

    # Preload ALL episode actions in one parquet scan
    data_dir = src_root / "data" / "chunk-000"
    parquet_file = str(sorted(data_dir.glob("*.parquet"))[0])
    all_actions = preload_all_actions(parquet_file)

    # Build env config
    env_cfg = parse_env_cfg(args.task, device=getattr(args, "device", "cpu"))
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path
    env_cfg.use_random_seed = False

    # Set initial garment (first in map)
    first_garment = ep_map[args.start_episode][0]
    env_cfg.garment_name = first_garment
    env_cfg.garment_version = garment_name_to_version(first_garment)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()  # required to init CUDA cloth physics view
    current_garment = first_garment

    meta = {}
    t_start = time.time()

    for ep_idx in range(args.start_episode, end):
        if not simulation_app.is_running():
            break

        garment_name, sub_ep_idx = ep_map[ep_idx]
        garment_type_str = garment_name_to_type_str(garment_name)
        type_idx = GARMENT_TYPE_TO_IDX.get(garment_type_str, 0)
        garment_version = garment_name_to_version(garment_name)

        # Switch garment if needed (switch_garment calls initialize_obs internally)
        if garment_name != current_garment:
            env.switch_garment(garment_name, garment_version)
            current_garment = garment_name
            print(f"  Switched to garment: {garment_name}")

        # Load initial pose for this sub-episode
        ep_info = garment_info[garment_name].get(str(sub_ep_idx), {})
        initial_pose = ep_info.get("object_initial_pose")

        # Reset env — use lightweight reset to avoid slow camera rendering
        indices = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        env._reset_idx(indices)
        env.scene.write_data_to_sim()
        env.sim.step(render=True)  # single render to sync USD state

        # Stabilize (render=False: fast, no kp read needed)
        zero_action = torch.zeros(1, 12, device=env.device)
        for _ in range(args.stabilize_steps):
            _physics_step(env, zero_action, render=False)

        # Set initial pose if available, then settle
        if initial_pose is not None:
            try:
                env.set_all_pose(initial_pose)
                for _ in range(10):
                    _physics_step(env, zero_action, render=False)
            except Exception as e:
                print(f"  [warn] Could not set initial pose for ep {ep_idx}: {e}")

        # Get preloaded actions for this episode
        if ep_idx not in all_actions:
            print(f"  [warn] No actions for ep {ep_idx}, skipping.")
            continue
        actions = all_actions[ep_idx]

        T = len(actions)
        kp_labels = np.full((T, 6, 2), -1.0, dtype=np.float32)

        for t in range(T):
            action_t = torch.from_numpy(actions[t]).float().unsqueeze(0).to(env.device)
            _physics_step(env, action_t)

            try:
                garment_object = env.object
                check_points = garment_object.check_points
                kp_3d_cm = get_object_particle_position(garment_object, check_points)
                kp_3d_arr = np.array(kp_3d_cm, dtype=np.float32)
                kp_2d = project_kp_3d_to_2d(kp_3d_arr)
                kp_labels[t] = kp_2d
                if t == 0:
                    print(f"  [debug] t=0: kp_3d_cm[0]={kp_3d_cm[0]}, kp_2d[0]={kp_2d[0]}", flush=True)
            except Exception as e:
                if t == 0:
                    print(f"  [warn] kp read failed at t=0: {e}", flush=True)
                pass  # leave as -1 (invalid)

        # Save episode keypoint labels
        out_path = output_dir / f"episode_{ep_idx:06d}.npy"
        np.save(str(out_path), kp_labels)

        meta[ep_idx] = {
            "garment_name": garment_name,
            "garment_type_str": garment_type_str,
            "type_idx": type_idx,
            "n_frames": T,
        }

        done = ep_idx - args.start_episode + 1
        total = end - args.start_episode
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] ep {ep_idx}: {garment_name} type={garment_type_str} T={T} "
              f"({rate:.2f} ep/s, ETA {eta/60:.0f}m)")

        if done % 10 == 0 or ep_idx == end - 1:
            with open(output_dir / "meta.json", "w") as f:
                json.dump({str(k): v for k, v in meta.items()}, f, indent=2)

    with open(output_dir / "meta.json", "w") as f:
        json.dump({str(k): v for k, v in meta.items()}, f, indent=2)

    env.close()
    print(f"\n[LabelKP] Done. Labels saved to {output_dir}")


def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import lehome.tasks.bedroom  # noqa: registers env
    label_all_episodes(args, simulation_app)
    simulation_app.close()


if __name__ == "__main__":
    main()
