"""Temporal Ensembling wrapper for LeRobot FoldFlow policy.

At every step, generates a fresh action chunk and maintains a buffer of the
last n_action_steps chunks. The action is a weighted average of all buffered
predictions for the current timestep, with exponential decay by chunk age.

  action_t = sum_k [ exp(-λ·k) · chunk_{t-k}[k] ] / sum_k exp(-λ·k)

where k=0 is the chunk just generated, k=1 is one step old, etc.
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


@PolicyRegistry.register("lerobot_te")
class TemporalEnsemblingLeRobotPolicy(LeRobotPolicy):
    """LeRobot FoldFlow policy with temporal ensembling at eval time.

    Generates a fresh action chunk every step and averages overlapping
    predictions with exponential decay weighting (decay=0.1 by default).
    """

    def __init__(self, *args, ensemble_decay: float = 0.1, replan_every: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_decay = ensemble_decay
        self.replan_every = replan_every  # regenerate chunk every N steps
        self._chunk_buffer: list = []  # newest chunk appended last
        self._step = 0
        self._last_chunk = None
        logger.info(f"[TE] Temporal ensembling enabled (decay={ensemble_decay}, replan_every={replan_every})")

    def reset(self):
        super().reset()
        self._chunk_buffer = []
        self._step = 0
        self._last_chunk = None

    def select_action(self, observation: Dict[str, Any]):
        fp = self.policy  # FoldFlowPolicy

        # Filter and preprocess (same as parent)
        if self.input_features:
            observation = self._filter_observations(observation, self.input_features)
        batch_obs = self._process_observation(observation)

        # Remove action key if present (FoldFlowPolicy expects this)
        batch_obs = {k: v for k, v in batch_obs.items() if k != ACTION}

        # Stack images the same way FoldFlowPolicy.select_action does
        if fp.config.image_features:
            batch_obs = dict(batch_obs)
            batch_obs[OBS_IMAGES] = torch.stack(
                [batch_obs[key] for key in fp.config.image_features], dim=-4
            )

        # Update observation queues in the underlying policy
        fp._queues = populate_queues(fp._queues, batch_obs)

        # Regenerate chunk every replan_every steps to limit compute
        if self._step % self.replan_every == 0:
            with torch.inference_mode():
                self._last_chunk = fp.predict_action_chunk(batch_obs)  # (1, horizon, action_dim)
            self._chunk_buffer.append(self._last_chunk)

        self._step += 1

        # Trim: only keep last n_action_steps chunks (older ones are out of range)
        n = fp.config.n_action_steps
        if len(self._chunk_buffer) > n:
            self._chunk_buffer = self._chunk_buffer[-n:]

        # Temporal ensemble: chunk generated k steps ago covers current step at index k
        k_decay = self.ensemble_decay
        action_sum = None
        total_weight = 0.0

        offset = self._step % self.replan_every  # steps since last replan

        for age_chunks, past_chunk in enumerate(reversed(self._chunk_buffer)):
            # age_chunks=0 → newest chunk, age_chunks=1 → one replan ago, ...
            action_idx = offset + age_chunks * self.replan_every
            if action_idx >= past_chunk.shape[1]:
                continue
            w = math.exp(-k_decay * age_chunks)
            action = past_chunk[:, action_idx, :]  # (1, action_dim)
            action_sum = w * action if action_sum is None else action_sum + w * action
            total_weight += w

        batch_action = action_sum / total_weight  # (1, action_dim)

        # Postprocess (unnormalize)
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)

        return batch_action.squeeze(0).cpu().numpy()
