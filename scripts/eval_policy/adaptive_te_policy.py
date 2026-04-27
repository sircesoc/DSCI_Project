"""Adaptive Temporal Ensembling — replans when action chunk confidence drops.

Instead of replanning every N steps, monitors the variance of the current
action chunk across the remaining horizon. When variance exceeds a threshold
(the policy is "unsure"), triggers an early replan.

This gives the best of both worlds:
- During smooth motions: executes longer without replanning (less jitter)
- During critical moments (grasping, folding): replans frequently (more reactive)
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


@PolicyRegistry.register("lerobot_ate")
class AdaptiveTELeRobotPolicy(LeRobotPolicy):
    """FoldFlow policy with adaptive temporal ensembling.

    Replans when:
    - Action chunk is exhausted (every max_replan steps)
    - OR the remaining actions in the chunk have high variance (uncertainty)
    """

    def __init__(self, *args, ensemble_decay: float = 0.1, min_replan: int = 2,
                 max_replan: int = 8, variance_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_decay = ensemble_decay
        self.min_replan = min_replan      # minimum steps between replans
        self.max_replan = max_replan      # maximum steps between replans
        self.variance_threshold = variance_threshold
        self._chunk_buffer: list = []
        self._step = 0
        self._steps_since_replan = 0
        self._current_chunk = None
        logger.info(f"[ATE] Adaptive TE: min_replan={min_replan}, max_replan={max_replan}, "
                    f"var_thresh={variance_threshold}, decay={ensemble_decay}")

    def reset(self):
        super().reset()
        self._chunk_buffer = []
        self._step = 0
        self._steps_since_replan = 0
        self._current_chunk = None

    def _should_replan(self) -> bool:
        """Decide whether to replan based on action chunk confidence."""
        # Always replan if we've hit max interval or have no chunk
        if self._current_chunk is None or self._steps_since_replan >= self.max_replan:
            return True

        # Don't replan if we just replanned
        if self._steps_since_replan < self.min_replan:
            return False

        # Check variance of remaining actions in current chunk
        remaining_start = self._steps_since_replan
        remaining = self._current_chunk[:, remaining_start:, :]  # (1, remaining_steps, action_dim)
        if remaining.shape[1] < 2:
            return True  # almost out of actions

        # Variance across the remaining horizon
        var = remaining.var(dim=1).mean().item()
        return var > self.variance_threshold

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

        # Adaptive replanning
        if self._should_replan():
            with torch.inference_mode():
                self._current_chunk = fp.predict_action_chunk(batch_obs)  # (1, horizon, action_dim)
            self._chunk_buffer.append(self._current_chunk)
            self._steps_since_replan = 0

        self._step += 1
        self._steps_since_replan += 1

        # Trim buffer
        n = fp.config.n_action_steps
        if len(self._chunk_buffer) > n:
            self._chunk_buffer = self._chunk_buffer[-n:]

        # Temporal ensemble across buffered chunks
        k_decay = self.ensemble_decay
        action_sum = None
        total_weight = 0.0

        # Calculate how many steps ago each chunk was generated
        cumulative_offset = self._steps_since_replan
        for age_chunks, past_chunk in enumerate(reversed(self._chunk_buffer)):
            if age_chunks == 0:
                action_idx = self._steps_since_replan - 1
            else:
                # Approximate: older chunks offset by their age
                action_idx = cumulative_offset + age_chunks * 4  # rough estimate
            if action_idx < 0 or action_idx >= past_chunk.shape[1]:
                continue
            w = math.exp(-k_decay * age_chunks)
            action = past_chunk[:, action_idx, :]
            action_sum = w * action if action_sum is None else action_sum + w * action
            total_weight += w

        if action_sum is None:
            # Fallback: use current chunk at current offset
            action_idx = min(self._steps_since_replan - 1, self._current_chunk.shape[1] - 1)
            batch_action = self._current_chunk[:, action_idx, :]
        else:
            batch_action = action_sum / total_weight

        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)

        return batch_action.squeeze(0).cpu().numpy()
