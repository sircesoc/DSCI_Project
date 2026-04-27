"""Residual RL policy wrapper for FoldFlow.

Wraps the base FoldFlow policy with a small residual MLP that learns
action corrections via REINFORCE during eval episodes.

The residual network is trained online — each episode contributes a
gradient update. Over many episodes, the residual learns to correct
the base policy's systematic mistakes (e.g., missed grasps).

Register as 'lerobot_rrl' to use with the standard eval pipeline.
"""

import math
from pathlib import Path
from typing import Dict, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.policies.utils import populate_queues

from lehome.utils.logger import get_logger
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


class ResidualMLP(nn.Module):
    """Small residual network that predicts per-step action corrections."""

    def __init__(self, obs_cond_dim: int, action_dim: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Initialize near-zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def forward(self, obs_cond: torch.Tensor):
        """Returns: (mean, std) for the residual action."""
        mean = self.net(obs_cond)  # (B, action_dim)
        std = self.log_std.exp()
        return mean, std


@PolicyRegistry.register("lerobot_rrl")
class ResidualRLPolicy(LeRobotPolicy):
    """FoldFlow + Temporal Ensembling + Online Residual RL.

    During eval, collects (obs_cond, residual, reward) tuples.
    After each episode, runs a REINFORCE update on the residual.
    """

    def __init__(self, *args, replan_every: int = 4, ensemble_decay: float = 0.1,
                 alpha: float = 0.1, rl_lr: float = 1e-4,
                 residual_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replan_every = replan_every
        self.ensemble_decay = ensemble_decay
        self.alpha = alpha

        # Get obs_cond_dim from the policy
        fp = self.policy
        obs_cond_dim = fp.model.dit.cond_proj.in_features - fp.model.dit.time_emb.dim
        action_dim = fp.config.action_feature.shape[0]

        # Create residual network
        self.residual = ResidualMLP(obs_cond_dim, action_dim)
        if residual_path and Path(residual_path).exists():
            self.residual.load_state_dict(torch.load(residual_path))
            logger.info(f"[RRL] Loaded residual from {residual_path}")

        self.optimizer = torch.optim.Adam(self.residual.parameters(), lr=rl_lr)

        # Episode buffers for RL
        self._episode_log_probs = []
        self._episode_rewards = []
        self._episode_count = 0

        # TE state
        self._chunk_buffer = []
        self._step = 0

        logger.info(f"[RRL] Residual RL: obs_cond_dim={obs_cond_dim}, alpha={alpha}, lr={rl_lr}")

    def reset(self):
        """Reset between episodes. Run REINFORCE update if we have data."""
        # REINFORCE update from previous episode
        if len(self._episode_rewards) > 0:
            self._reinforce_update()
            self._episode_count += 1

        super().reset()
        self._chunk_buffer = []
        self._step = 0
        self._episode_log_probs = []
        self._episode_rewards = []

    def _reinforce_update(self):
        """Run one REINFORCE gradient step from the completed episode."""
        rewards = self._episode_rewards
        log_probs = self._episode_log_probs

        if len(rewards) == 0 or len(log_probs) == 0:
            return

        # Compute discounted returns
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = 0
        n = min(len(log_probs), len(returns))
        for i in range(n):
            loss -= log_probs[i] * returns[i]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.residual.parameters(), 1.0)
        self.optimizer.step()

        ep_return = sum(rewards)
        logger.info(f"[RRL] Episode {self._episode_count}: return={ep_return:.2f}, "
                    f"steps={len(rewards)}, loss={loss.item():.4f}")

        # Save periodically
        if self._episode_count > 0 and self._episode_count % 20 == 0:
            save_path = Path("outputs/residual_rl")
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.residual.state_dict(),
                      save_path / f"residual_ep{self._episode_count}.pt")
            logger.info(f"[RRL] Saved checkpoint at episode {self._episode_count}")

    def select_action(self, observation: Dict[str, Any]):
        fp = self.policy

        # Filter and preprocess
        if self.input_features:
            observation = self._filter_observations(observation, self.input_features)
        batch_obs = self._process_observation(observation)
        batch_obs = {k: v for k, v in batch_obs.items() if k != ACTION}

        if fp.config.image_features:
            batch_obs = dict(batch_obs)
            batch_obs[OBS_IMAGES] = torch.stack(
                [batch_obs[key] for key in fp.config.image_features], dim=-4
            )

        fp._queues = populate_queues(fp._queues, batch_obs)

        # Generate base action chunk via TE
        if self._step % self.replan_every == 0:
            with torch.inference_mode():
                chunk = fp.predict_action_chunk(batch_obs)
            self._chunk_buffer.append(chunk)

        self._step += 1

        # Trim buffer
        n = fp.config.n_action_steps
        if len(self._chunk_buffer) > n:
            self._chunk_buffer = self._chunk_buffer[-n:]

        # TE averaging for base action
        k_decay = self.ensemble_decay
        action_sum = None
        total_weight = 0.0
        offset = self._step % self.replan_every

        for age_chunks, past_chunk in enumerate(reversed(self._chunk_buffer)):
            action_idx = offset + age_chunks * self.replan_every
            if action_idx >= past_chunk.shape[1]:
                continue
            w = math.exp(-k_decay * age_chunks)
            action = past_chunk[:, action_idx, :]
            action_sum = w * action if action_sum is None else action_sum + w * action
            total_weight += w

        base_action = action_sum / total_weight  # (1, action_dim)

        # Get obs_cond for residual
        with torch.no_grad():
            batch_for_encode = {}
            for k, q in fp._queues.items():
                if k != "action" and len(q) > 0:
                    batch_for_encode[k] = torch.stack(list(q), dim=1)
            if batch_for_encode:
                obs_cond = fp.model._encode_obs(batch_for_encode)
            else:
                obs_cond = torch.zeros(1, self.residual.net[0].in_features)

        # Compute residual with exploration noise
        obs_cond_cpu = obs_cond.detach().cpu()
        mean, std = self.residual(obs_cond_cpu)
        noise = torch.randn_like(mean)
        residual_action = mean + std * noise

        # Log prob for REINFORCE
        log_prob = -0.5 * ((residual_action - mean) / std).pow(2).sum()
        log_prob -= 0.5 * self.residual.action_dim * math.log(2 * math.pi)
        log_prob -= self.residual.log_std.sum()
        self._episode_log_probs.append(log_prob)

        # Combine base + residual (ensure same device)
        final_action = base_action.cpu() + self.alpha * residual_action

        # Store reward (will be filled by environment's reward signal)
        # For now, use a simple heuristic: did the action change much from base?
        # Real reward will come from _get_rewards() in the eval loop
        self._episode_rewards.append(0.0)  # placeholder, updated externally

        # Postprocess
        if self.postprocessor:
            final_action = self.postprocessor(final_action)

        return final_action.squeeze(0).detach().cpu().numpy()

    def add_reward(self, reward: float):
        """Called externally to provide reward for the last action."""
        if self._episode_rewards:
            self._episode_rewards[-1] = reward
