"""PhaseGatedLeRobotPolicy
========================
Wraps LeRobotPolicy with a phase-aware state machine for hierarchical
fold planning (v5).

Tracks fold phases via gripper cycle detection.  When a cycle completes
(the robot finishes a fold attempt), checks 2D keypoint sub-goals.  If
the sub-goal is not met the phase is held and the inner policy is forced
to re-plan at the same phase.

Always injects ``observation.phase`` into the observation dict so that
phase-conditioned models can condition on the current fold phase.

Registration
------------
Registered as ``"lerobot_phase_gated"`` in the policy registry.
Use ``--policy_type lerobot_phase_gated`` when calling eval.py.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from lehome.utils.logger import get_logger
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Gripper cycle detector (state-space thresholds for eval)
# --------------------------------------------------------------------------

class GripperCycleDetector:
    """Detects a full grasp→release cycle on a single gripper.

    Thresholds calibrated for state-space values at eval time:
    - Closed (grasping): value < close_thresh  (~0.35)
    - Open   (released): value > open_thresh   (~0.60)
    """

    def __init__(self, close_thresh: float = 0.35, open_thresh: float = 0.60,
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
        """Feed one gripper reading.  Returns True on completed cycle."""
        if value < self.close_thresh:
            self._close_steps += 1
            if self._close_steps >= self.min_close_steps:
                self._was_closed = True
        else:
            self._close_steps = 0

        if self._was_closed and value > self.open_thresh:
            self._was_closed = False
            self._close_steps = 0
            return True
        return False


# --------------------------------------------------------------------------
# Phase sub-goal checker (2D keypoint distances)
# --------------------------------------------------------------------------

# Per garment-type index (0-3) and phase (0-1): keypoint pairs that must
# be "close" after a successful fold.  Phase 2 is terminal — always passes.
_PHASE_SUBGOALS: dict[int, dict[int, list[tuple[int, int]]]] = {
    0: {0: [(0, 4)], 1: [(1, 5)]},  # top-long-sleeve
    1: {0: [(0, 4)], 1: [(1, 5)]},  # top-short-sleeve
    2: {0: [(0, 4)], 1: [(1, 5)]},  # long-pant
    3: {0: [(0, 1)], 1: [(4, 5)]},  # short-pant
}


def _check_phase_subgoal(
    kp_2d: np.ndarray,
    garment_type_idx: int,
    phase: int,
    close_thresh: float,
) -> bool:
    """Check if a phase's fold sub-goal is met via 2D keypoint distances.

    Args:
        kp_2d:             (6, 2) normalised keypoints in [0, 1].
        garment_type_idx:  0-3 garment type index.
        phase:             0-2 fold phase to check.
        close_thresh:      Max normalised distance for "close" criterion.

    Returns:
        True if sub-goal is met (or phase is terminal / uncheckable).
    """
    if phase >= 2:
        return True

    pairs = _PHASE_SUBGOALS.get(garment_type_idx, {}).get(phase)
    if pairs is None:
        return True

    for kp_a, kp_b in pairs:
        if kp_2d[kp_a, 0] < 0 or kp_2d[kp_b, 0] < 0:
            continue  # keypoint unavailable — skip
        dist = float(np.linalg.norm(kp_2d[kp_a] - kp_2d[kp_b]))
        if dist > close_thresh:
            return False

    return True


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11
MAX_PHASE = 2
N_ACTION_STEPS = 16  # action chunk size


# --------------------------------------------------------------------------
# Phase-gated policy
# --------------------------------------------------------------------------

@PolicyRegistry.register("lerobot_phase_gated")
class PhaseGatedLeRobotPolicy(LeRobotPolicy):
    """LeRobotPolicy with phase-gated execution for hierarchical folding.

    Two-track phase tracking:

    * **gripper_phase** — incremented whenever a gripper close→open cycle
      is detected (the robot *attempted* a fold).
    * **gated_phase** — the phase actually injected into ``observation.phase``.
      Only advances when the keypoint sub-goal for the current phase is met
      (or max retries are exhausted).

    At each action-chunk boundary, if gripper_phase > gated_phase the
    wrapper checks the 2D keypoint sub-goal.  On failure the inner policy
    is reset (forcing a fresh re-plan at the same phase).

    Args:
        max_retries_per_phase:  Re-plans allowed per phase before force-advance.
        subgoal_close_thresh:   Normalised 2D distance threshold (~0.10).
        close_thresh:           Gripper cycle "closed" threshold (state space).
        open_thresh:            Gripper cycle "open" threshold (state space).
    """

    def __init__(
        self,
        *args,
        max_retries_per_phase: int = 3,
        subgoal_close_thresh: float = 0.10,
        close_thresh: float = 0.35,
        open_thresh: float = 0.60,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries_per_phase
        self.subgoal_thresh = subgoal_close_thresh

        self._left_det = GripperCycleDetector(close_thresh, open_thresh)
        self._right_det = GripperCycleDetector(close_thresh, open_thresh)

        # Phase state
        self._gripper_phase = 0   # inferred from gripper cycles
        self._gated_phase = 0     # actually injected into obs
        self._retries = 0
        self._chunk_step = 0
        self._garment_type_idx = 0
        self._pending_check = False  # set when a gripper cycle fires

    def reset(self):
        super().reset()
        self._left_det.reset()
        self._right_det.reset()
        self._gripper_phase = 0
        self._gated_phase = 0
        self._retries = 0
        self._chunk_step = 0
        self._garment_type_idx = 0
        self._pending_check = False
        logger.info("[PhaseGated] Episode reset — phase=0")

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        state = observation.get("observation.state")
        kp_2d = observation.get("observation.keypoints")

        # Track garment type from eval-injected scalar
        gt = observation.get("observation.garment_type")
        if gt is not None:
            self._garment_type_idx = int(round(float(np.asarray(gt).flat[0])))

        # --- Gripper cycle detection ---
        if state is not None:
            left_cycle = self._left_det.update(float(state[LEFT_GRIPPER_IDX]))
            right_cycle = self._right_det.update(float(state[RIGHT_GRIPPER_IDX]))
            if left_cycle or right_cycle:
                self._gripper_phase = min(self._gripper_phase + 1, MAX_PHASE)
                self._pending_check = True
                side = "left" if left_cycle else "right"
                logger.info(
                    f"[PhaseGated] {side} gripper cycle → "
                    f"gripper_phase={self._gripper_phase}"
                )

        # --- Sub-goal check at chunk boundaries ---
        self._chunk_step += 1
        if self._chunk_step >= N_ACTION_STEPS:
            self._chunk_step = 0

            if self._pending_check and self._gripper_phase > self._gated_phase:
                self._pending_check = False

                if kp_2d is not None:
                    subgoal_met = _check_phase_subgoal(
                        kp_2d, self._garment_type_idx,
                        self._gated_phase, self.subgoal_thresh,
                    )
                else:
                    subgoal_met = True  # no keypoints available → pass

                if subgoal_met:
                    old = self._gated_phase
                    self._gated_phase = self._gripper_phase
                    self._retries = 0
                    logger.info(
                        f"[PhaseGated] Phase {old} sub-goal MET → "
                        f"advance to phase {self._gated_phase}"
                    )
                elif self._retries < self.max_retries:
                    self._retries += 1
                    # Revert gripper phase and reset detectors for fresh cycle
                    self._gripper_phase = self._gated_phase
                    self._left_det.reset()
                    self._right_det.reset()
                    # Force inner policy to re-plan
                    self.policy.reset()
                    logger.info(
                        f"[PhaseGated] Phase {self._gated_phase} sub-goal NOT met → "
                        f"re-plan (retry {self._retries}/{self.max_retries})"
                    )
                else:
                    old = self._gated_phase
                    self._gated_phase = self._gripper_phase
                    self._retries = 0
                    logger.warning(
                        f"[PhaseGated] Phase {old} max retries exhausted → "
                        f"force advance to phase {self._gated_phase}"
                    )

        # --- Inject observation.phase ---
        observation["observation.phase"] = np.array(
            [float(self._gated_phase)], dtype=np.float32,
        )

        return super().select_action(observation)
