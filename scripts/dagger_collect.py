#!/usr/bin/env python
"""DAgger data collection: policy runs autonomously, human takes over on failure.

Usage:
    python -m scripts.dagger_collect \
        --policy_path outputs/train/foldflow_v8b/checkpoints/300000/pretrained_model \
        --dataset_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged \
        --teleop_device bi-so101leader \
        --left_arm_port /dev/ttyACM0 \
        --right_arm_port /dev/ttyACM1 \
        --garment_type top_short \
        --garment_name Top_Short_Seen_8 \
        --num_episodes 25 \
        --device cpu

Controls during recording:
    S — Start recording
    T — Take over (human controls via SO-101 arms)
    R — Release (policy resumes control)
    N — Save episode
    D — Discard and re-record
    ESC — Abort session
"""

import argparse
import sys
from pathlib import Path

# Isaac Sim must be imported before anything else
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.devices import BiSO101Leader
from lehome.utils.record import get_next_experiment_path_with_gap
from lehome.utils.logger import get_logger

# Import policy
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.lerobot_policy import LeRobotPolicy
from scripts.utils.dagger_record import run_dagger_recording

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DAgger data collection")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to trained FoldFlow checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset root (for policy metadata)")
    parser.add_argument("--teleop_device", type=str, default="bi-so101leader")
    parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--garment_type", type=str, default="top_short",
                        choices=["top_long", "top_short", "pant_long", "pant_short"])
    parser.add_argument("--garment_name", type=str, default=None,
                        help="Specific garment variant name (e.g., Top_Short_Seen_8)")
    parser.add_argument("--num_episodes", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="Datasets/dagger",
                        help="Output directory for DAgger episodes")
    parser.add_argument("--task", type=str, default="LeHome-BiSO101-Direct-Garment-v2")
    parser.add_argument("--step_hz", type=int, default=120)
    parser.add_argument("--disable_depth", action="store_true")
    args = parser.parse_args()

    args.task_description = f"Fold {args.garment_type} garment"
    args.num_episode = args.num_episodes

    # 1. Create environment (same pattern as dataset_record.py)
    logger.info("Creating environment...")
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.garment_name = args.garment_name
    env_cfg.garment_version = getattr(args, "garment_version", "Release")
    env_cfg.garment_cfg_base_path = getattr(args, "garment_cfg_base_path", "Assets/objects/Challenge_Garment")
    env_cfg.particle_cfg_path = getattr(args, "particle_cfg_path", "Assets/objects/Challenge_Garment/particle_cfg.yaml")
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # 2. Load trained policy
    logger.info(f"Loading policy from {args.policy_path}")
    policy = LeRobotPolicy(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        task_description=args.task_description,
        device=args.device,
    )

    # 3. Create teleop interface
    logger.info("Connecting to SO-101 leader arms...")
    teleop = BiSO101Leader(
        env,
        left_port=args.left_arm_port,
        right_port=args.right_arm_port,
        recalibrate=args.recalibrate,
    )

    # 4. Create dataset for recording
    output_path = get_next_experiment_path_with_gap(Path(args.output_dir))
    logger.info(f"Recording to: {output_path}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (12,)},
        "action": {"dtype": "float32", "shape": (12,)},
        "observation.images.top_rgb": {"dtype": "video", "shape": (480, 640, 3), "video_info": {"fps": 30}},
        "observation.images.left_rgb": {"dtype": "video", "shape": (480, 640, 3), "video_info": {"fps": 30}},
        "observation.images.right_rgb": {"dtype": "video", "shape": (480, 640, 3), "video_info": {"fps": 30}},
    }

    dataset = LeRobotDataset.create(
        repo_id=f"dagger_{args.garment_type}",
        root=str(output_path),
        fps=30,
        image_writer_threads=8,
        features=features,
    )

    json_path = output_path / "meta" / "garment_info.json"

    # 5. Run DAgger recording
    logger.info("=" * 60)
    logger.info("DAgger Data Collection")
    logger.info(f"  Policy: {args.policy_path}")
    logger.info(f"  Garment: {args.garment_name or args.garment_type}")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info("  Controls: S=start, T=takeover, R=release, N=save, D=discard")
    logger.info("=" * 60)

    try:
        run_dagger_recording(
            env=env,
            teleop_interface=teleop,
            policy=policy,
            args=args,
            dataset=dataset,
            json_path=json_path,
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
