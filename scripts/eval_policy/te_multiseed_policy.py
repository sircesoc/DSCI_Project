"""Temporal Ensembling + Multi-Seed wrapper for FoldFlow policy.

Combines two eval-time improvements:
1. Temporal ensembling (replan every N steps, average overlapping chunks)
2. Multi-seed inference (generate K chunks with different noise seeds,
   pick the one with lowest variance across the action horizon)

The lowest-variance chunk is the most "confident" prediction — it means
the flow matching ODE converged to a consistent trajectory regardless of
the initial noise sample.
"""

import math
from typing import Dict, Any

import torch

from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.policies.utils import populate_queues

from lehome.utils.logger import get_logger
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


@PolicyRegistry.register("lerobot_te_ms")
class TEMultiSeedLeRobotPolicy(LeRobotPolicy):
    """FoldFlow policy with temporal ensembling + multi-seed confidence selection.

    At each replan step:
    1. Generate n_seeds action chunks with different noise seeds
    2. Pick the chunk with lowest variance (most confident)
    3. Average with previous chunks via temporal ensembling
    """

    def __init__(self, *args, ensemble_decay: float = 0.1, replan_every: int = 4,
                 n_seeds: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_decay = ensemble_decay
        self.replan_every = replan_every
        self.n_seeds = n_seeds
        self._chunk_buffer: list = []
        self._step = 0
        logger.info(f"[TE+MS] decay={ensemble_decay}, replan_every={replan_every}, n_seeds={n_seeds}")

    def reset(self):
        super().reset()
        self._chunk_buffer = []
        self._step = 0

    def _generate_best_chunk(self, batch_obs: dict) -> torch.Tensor:
        """Generate n_seeds chunks and return the one with lowest variance."""
        fp = self.policy
        chunks = []
        for _ in range(self.n_seeds):
            with torch.inference_mode():
                chunk = fp.predict_action_chunk(batch_obs)  # (1, horizon, action_dim)
            chunks.append(chunk)

        if self.n_seeds == 1:
            return chunks[0]

        # Pick lowest variance chunk (most confident prediction)
        stacked = torch.stack(chunks, dim=0)  # (n_seeds, 1, horizon, action_dim)
        # Variance across the action horizon for each seed
        variances = stacked.var(dim=2).mean(dim=(1, 2))  # (n_seeds,)
        best_idx = variances.argmin().item()
        return chunks[best_idx]

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

        # Regenerate chunk every replan_every steps
        if self._step % self.replan_every == 0:
            best_chunk = self._generate_best_chunk(batch_obs)
            self._chunk_buffer.append(best_chunk)

        self._step += 1

        # Trim buffer
        n = fp.config.n_action_steps
        if len(self._chunk_buffer) > n:
            self._chunk_buffer = self._chunk_buffer[-n:]

        # Temporal ensemble
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

        batch_action = action_sum / total_weight

        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)

        return batch_action.squeeze(0).cpu().numpy()
