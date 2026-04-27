#!/usr/bin/env python
"""Residual RL fine-tuning for FoldFlow.

Trains a small residual network on top of the frozen v8b policy using
REINFORCE with keypoint-based rewards in Isaac Sim.

Architecture:
    base_action = frozen_v8b.generate_actions(obs)      # (B, horizon, 12)
    residual    = residual_net(obs_cond)                 # (B, horizon, 12)
    final_action = base_action + alpha * residual        # alpha starts small

The residual learns to correct the base policy's mistakes without
degrading its strengths. Only the residual network is trained.

Usage:
    python -m scripts.residual_rl \
        --policy_path outputs/train/foldflow_v8b/checkpoints/300000/pretrained_model \
        --dataset_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged \
        --garment_type top_short \
        --num_episodes 200 \
        --device cpu
"""

import argparse
import sys
from pathlib import Path

# Isaac Sim must be imported first
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from lehome.utils.logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.eval_policy.lerobot_policy import LeRobotPolicy
from scripts.utils.common import stabilize_garment_after_reset

logger = get_logger(__name__)

# Garment type mapping
GARMENT_TYPE_MAP = {
    "top_long": ("LeHome-BiSO101-Direct-Garment-v2", "top-long-sleeve", 0),
    "top_short": ("LeHome-BiSO101-Direct-Garment-v2", "top-short-sleeve", 1),
    "pant_long": ("LeHome-BiSO101-Direct-Garment-v2", "long-pant", 2),
    "pant_short": ("LeHome-BiSO101-Direct-Garment-v2", "short-pant", 3),
}


class ResidualPolicy(nn.Module):
    """Small residual MLP that predicts action corrections.

    Input: obs_cond from FoldFlow's vision encoder (obs_cond_dim)
    Output: residual action chunk (horizon, action_dim)
    """

    def __init__(self, obs_cond_dim: int, horizon: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

        # Initialize near-zero so residual starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Log std for action noise (learnable)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def forward(self, obs_cond: torch.Tensor):
        """Predict residual action and sample with noise for exploration.

        Args:
            obs_cond: (B, obs_cond_dim) encoded observation.

        Returns:
            residual: (B, horizon, action_dim) sampled residual.
            log_prob: (B,) log probability for REINFORCE.
        """
        B = obs_cond.shape[0]
        mean = self.net(obs_cond).reshape(B, self.horizon, self.action_dim)
        std = self.log_std.exp().unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)

        # Sample from Gaussian
        noise = torch.randn_like(mean)
        residual = mean + std * noise

        # Log probability
        log_prob = -0.5 * ((residual - mean) / std).pow(2).sum(dim=(1, 2))
        log_prob -= 0.5 * self.horizon * self.action_dim * np.log(2 * np.pi)
        log_prob -= self.log_std.sum() * self.horizon

        return residual, log_prob


def compute_keypoint_reward(env) -> float:
    """Compute reward based on keypoint distances toward folded state.

    Uses the success checker's keypoint distance metrics.
    """
    try:
        from lehome.utils.success_checker_chanllege import (
            get_object_particle_position,
            check_garment_success,
        )
        garment_object = env.object
        check_points = garment_object.check_points
        kp_3d = get_object_particle_position(garment_object, check_points)

        # Compute fold distances (same pairs as success checker)
        kp = np.array(kp_3d, dtype=np.float32)
        d04 = np.linalg.norm(kp[0] - kp[4])
        d15 = np.linalg.norm(kp[1] - kp[5])
        d23 = np.linalg.norm(kp[2] - kp[3])

        # Negative fold distance as reward (lower distance = higher reward)
        fold_dist = d04 + d15 + d23
        return -fold_dist
    except Exception:
        return 0.0


def rollout_episode(
    env: DirectRLEnv,
    base_policy: LeRobotPolicy,
    residual: ResidualPolicy,
    alpha: float,
    garment_type_idx: int,
    max_steps: int = 600,
    device: str = "cpu",
) -> dict:
    """Run one episode with base policy + residual corrections.

    Returns:
        Dictionary with episode data: rewards, log_probs, success, length.
    """
    env.reset()
    stabilize_garment_after_reset(env, argparse.Namespace(step_hz=120))
    base_policy.reset()

    observation_dict = env._get_observations()
    observation_dict["observation.garment_type"] = np.array(
        garment_type_idx, dtype=np.float32
    )

    rewards = []
    log_probs = []
    success = False

    # Get initial keypoint reward for baseline
    prev_kp_reward = compute_keypoint_reward(env)

    for step in range(max_steps):
        # Get base action from frozen policy
        action_np = base_policy.select_action(observation_dict)
        base_action = torch.from_numpy(action_np).float().unsqueeze(0)  # (1, 12)

        # Get obs_cond from the frozen policy's vision encoder
        fp = base_policy.policy  # FoldFlowPolicy
        with torch.no_grad():
            # Build batch from queues
            batch = {}
            for k, q in fp._queues.items():
                if k != "action" and len(q) > 0:
                    batch[k] = torch.stack(list(q), dim=1)
            if batch:
                obs_cond = fp.model._encode_obs(batch)  # (1, obs_cond_dim)
            else:
                obs_cond = torch.zeros(1, 512, dtype=torch.float32)  # fallback

        # Get residual correction
        residual_action, log_prob = residual(obs_cond.detach())
        # Only take the first action step from the residual chunk
        residual_step = residual_action[:, 0, :]  # (1, 12)

        # Combine: base + scaled residual
        final_action = base_action + alpha * residual_step
        final_action = final_action.to(env.device)

        # Step environment
        env.step(final_action)

        # Compute reward
        curr_kp_reward = compute_keypoint_reward(env)
        step_reward = curr_kp_reward - prev_kp_reward  # positive when improving
        prev_kp_reward = curr_kp_reward

        # Check success
        success_tensor = env._get_success()
        if success_tensor.item():
            step_reward += 10.0  # bonus for success
            success = True

        rewards.append(step_reward)
        log_probs.append(log_prob)

        # Update observation
        observation_dict = env._get_observations()
        observation_dict["observation.garment_type"] = np.array(
            garment_type_idx, dtype=np.float32
        )

        _, truncated = env._get_dones()
        if truncated or success:
            break

    return {
        "rewards": rewards,
        "log_probs": log_probs,
        "success": success,
        "length": len(rewards),
        "total_return": sum(rewards),
    }


def train_residual_rl(
    env: DirectRLEnv,
    base_policy: LeRobotPolicy,
    garment_type_idx: int,
    num_episodes: int = 200,
    lr: float = 1e-4,
    alpha_start: float = 0.05,
    alpha_end: float = 0.2,
    gamma: float = 0.99,
    save_every: int = 50,
    save_dir: str = "outputs/residual_rl",
    device: str = "cpu",
):
    """Train residual policy with REINFORCE."""

    # Determine obs_cond_dim from the base policy
    fp = base_policy.policy
    obs_cond_dim = fp.model.dit.cond_proj.in_features - fp.model.dit.time_emb.dim
    horizon = fp.config.n_action_steps
    action_dim = fp.config.action_feature.shape[0]

    logger.info(f"Residual RL: obs_cond_dim={obs_cond_dim}, horizon={horizon}, action_dim={action_dim}")

    residual = ResidualPolicy(obs_cond_dim, horizon, action_dim).to(device)
    optimizer = torch.optim.Adam(residual.parameters(), lr=lr)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Tracking
    recent_returns = deque(maxlen=20)
    recent_successes = deque(maxlen=20)
    best_success_rate = 0.0

    for ep in range(num_episodes):
        # Anneal alpha: start small, grow as residual improves
        alpha = alpha_start + (alpha_end - alpha_start) * min(1.0, ep / (num_episodes * 0.5))

        # Rollout
        result = rollout_episode(
            env, base_policy, residual, alpha, garment_type_idx,
            device=device,
        )

        recent_returns.append(result["total_return"])
        recent_successes.append(float(result["success"]))

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(result["rewards"]):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE loss
        policy_loss = 0
        for log_prob, G in zip(result["log_probs"], returns):
            policy_loss -= log_prob * G

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
        optimizer.step()

        # Logging
        avg_return = np.mean(recent_returns)
        success_rate = np.mean(recent_successes)

        if (ep + 1) % 10 == 0:
            logger.info(
                f"Episode {ep+1}/{num_episodes} | "
                f"Return: {result['total_return']:.2f} | "
                f"Avg Return: {avg_return:.2f} | "
                f"Success: {result['success']} | "
                f"Success Rate: {success_rate*100:.0f}% | "
                f"Alpha: {alpha:.3f} | "
                f"Length: {result['length']}"
            )

        # Save best
        if success_rate > best_success_rate and len(recent_successes) >= 10:
            best_success_rate = success_rate
            torch.save(residual.state_dict(), save_path / "residual_best.pt")
            logger.info(f"New best success rate: {success_rate*100:.0f}%")

        # Periodic save
        if (ep + 1) % save_every == 0:
            torch.save(residual.state_dict(), save_path / f"residual_ep{ep+1}.pt")
            logger.info(f"Checkpoint saved at episode {ep+1}")

    # Final save
    torch.save(residual.state_dict(), save_path / "residual_final.pt")
    logger.info(f"Training complete. Best success rate: {best_success_rate*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Residual RL for FoldFlow")
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--garment_type", type=str, default="top_short",
                        choices=["top_long", "top_short", "pant_long", "pant_short"])
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha_start", type=float, default=0.05)
    parser.add_argument("--alpha_end", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="outputs/residual_rl")
    args = parser.parse_args()

    task_name, garment_type_str, garment_type_idx = GARMENT_TYPE_MAP[args.garment_type]

    # Create environment
    logger.info("Creating environment...")
    env_cfg = parse_env_cfg(task_name)
    env = DirectRLEnv(cfg=env_cfg)

    # Load frozen base policy
    logger.info(f"Loading base policy from {args.policy_path}")
    base_policy = LeRobotPolicy(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        task_description=f"Fold {args.garment_type} garment",
        device=args.device,
    )

    logger.info("=" * 60)
    logger.info("Residual RL Training")
    logger.info(f"  Base policy: {args.policy_path}")
    logger.info(f"  Garment type: {args.garment_type}")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info(f"  Alpha: {args.alpha_start} → {args.alpha_end}")
    logger.info("=" * 60)

    try:
        train_residual_rl(
            env=env,
            base_policy=base_policy,
            garment_type_idx=garment_type_idx,
            num_episodes=args.num_episodes,
            lr=args.lr,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
            save_dir=args.save_dir,
            device=args.device,
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
