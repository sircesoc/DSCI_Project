"""
RecoveryLeRobotPolicy
=====================
Extends LeRobotPolicy with phase-aware retry logic for clothes folding.

Problem
-------
FoldFlow (and any imitation policy) plans a 32-step action chunk that implicitly
assumes earlier phases succeeded. If the first sleeve fold fails, the policy keeps
advancing into the second-sleeve phase even though the garment is still unfolded.

Solution
--------
Track folding phases by monitoring gripper open→close→open cycles in the robot
state. Each complete grasp-release cycle = one fold phase completed. If a phase
runs too long without a detected cycle (timeout), force a re-plan from the current
observation so the policy retries that phase from scratch.

Registration
------------
Registered as "lerobot_recovery" in the policy registry.
Use --policy_type lerobot_recovery when calling eval.py.
"""

from typing import Dict

import numpy as np

from lehome.utils.logger import get_logger
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Gripper cycle detector
# --------------------------------------------------------------------------

class GripperCycleDetector:
    """Detects a full grasp-release cycle on a single gripper.

    A cycle is: value drops below `close_thresh` (grasp) then rises above
    `open_thresh` (release). The detector latches the closed state until the
    gripper opens again.

    Args:
        close_thresh: Normalised gripper value below which we consider it closed.
        open_thresh:  Normalised gripper value above which we consider it open.
        min_close_steps: Minimum consecutive steps below close_thresh before
                         we declare a grasp (avoids transient noise).
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


# --------------------------------------------------------------------------
# Recovery policy
# --------------------------------------------------------------------------

@PolicyRegistry.register("lerobot_recovery")
class RecoveryLeRobotPolicy(LeRobotPolicy):
    """LeRobotPolicy with per-phase retry for clothes-folding tasks.

    Phase model (top_long has 3 gripper-cycle phases):
        Phase 0 → first sleeve fold  (left or right gripper cycle #1)
        Phase 1 → second sleeve fold (gripper cycle #2)
        Phase 2 → final body fold    (gripper cycle #3)
        Phase 3 → done

    If `phase_timeout` steps pass without the expected cycle, the policy
    action queue is cleared and a new chunk is generated from the current
    observation (retry). After `max_retries` failed attempts the policy
    gives up on retrying that phase and lets execution continue naturally.

    Args:
        max_retries:        Max retry attempts per phase before giving up.
        phase_timeout:      Steps per phase before declaring failure.
        close_thresh:       Gripper value considered closed/grasping.
        open_thresh:        Gripper value considered open/released.
        min_close_steps:    Min consecutive close readings to register a grasp.
        movement_threshold: Minimum L2 state change over a chunk to count as
                            "meaningful movement" (secondary guard).
    """

    # State vector indices for the two grippers
    LEFT_GRIPPER_IDX = 5
    RIGHT_GRIPPER_IDX = 11
    # Number of gripper-cycle phases expected for top_long folding
    NUM_PHASES = 3

    def __init__(
        self,
        *args,
        max_retries: int = 3,
        phase_timeout: int = 100,
        close_thresh: float = 0.35,
        open_thresh: float = 0.60,
        min_close_steps: int = 3,
        movement_threshold: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.phase_timeout = phase_timeout
        self.movement_threshold = movement_threshold

        self._left_detector = GripperCycleDetector(close_thresh, open_thresh, min_close_steps)
        self._right_detector = GripperCycleDetector(close_thresh, open_thresh, min_close_steps)

        # Episode-level counters (reset each episode)
        self._phase = 0
        self._retry_count = 0
        self._steps_in_phase = 0
        self._step_total = 0

        # State snapshot at the start of the most recent planning chunk
        self._state_at_replan: np.ndarray | None = None
        # Step counter within the current action chunk (0…n_action_steps-1)
        self._chunk_step = 0
        self._n_action_steps = getattr(
            getattr(self, "policy", None), "config", None
        )
        if self._n_action_steps is not None:
            self._n_action_steps = getattr(self._n_action_steps, "n_action_steps", 16)
        else:
            self._n_action_steps = 16

    def reset(self):
        super().reset()
        self._left_detector.reset()
        self._right_detector.reset()
        self._phase = 0
        self._retry_count = 0
        self._steps_in_phase = 0
        self._step_total = 0
        self._state_at_replan = None
        self._chunk_step = 0
        logger.info("RecoveryLeRobotPolicy: episode reset")

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        state = observation.get("observation.state", None)

        # ----------------------------------------------------------------
        # 1. Detect gripper cycles → phase advancement
        # ----------------------------------------------------------------
        if state is not None and self._phase < self.NUM_PHASES:
            left_val = float(state[self.LEFT_GRIPPER_IDX])
            right_val = float(state[self.RIGHT_GRIPPER_IDX])

            left_cycle = self._left_detector.update(left_val)
            right_cycle = self._right_detector.update(right_val)

            if left_cycle or right_cycle:
                side = "left" if left_cycle else "right"
                logger.info(
                    f"[Recovery] {side} gripper cycle detected → "
                    f"phase {self._phase} complete at step {self._step_total}"
                )
                self._phase += 1
                self._retry_count = 0
                self._steps_in_phase = 0
                self._state_at_replan = None  # fresh snapshot on next chunk

        # ----------------------------------------------------------------
        # 2. Check for phase timeout → force retry
        # ----------------------------------------------------------------
        if state is not None and self._phase < self.NUM_PHASES:
            self._steps_in_phase += 1

            if self._steps_in_phase >= self.phase_timeout:
                if self._retry_count < self.max_retries:
                    self._retry_count += 1
                    logger.warning(
                        f"[Recovery] Phase {self._phase} timed out after "
                        f"{self._steps_in_phase} steps "
                        f"(retry {self._retry_count}/{self.max_retries}). "
                        f"Forcing re-plan from current state."
                    )
                    # Clear the inner policy's action AND observation queues
                    # so it generates a fresh chunk from the current observation.
                    self.policy.reset()
                    # Reset cycle detectors so we don't double-count
                    self._left_detector.reset()
                    self._right_detector.reset()
                    self._steps_in_phase = 0
                    self._chunk_step = 0
                    self._state_at_replan = None
                else:
                    # Exhausted retries — give up and let execution continue
                    if self._steps_in_phase == self.phase_timeout:
                        logger.warning(
                            f"[Recovery] Phase {self._phase} retries exhausted "
                            f"({self.max_retries}). Continuing without retry."
                        )

        # ----------------------------------------------------------------
        # 3. Snapshot state at the start of each new action chunk
        # ----------------------------------------------------------------
        if self._chunk_step == 0 and state is not None:
            self._state_at_replan = state.copy()

        # ----------------------------------------------------------------
        # 4. Secondary guard: movement check at end of chunk
        # ----------------------------------------------------------------
        self._chunk_step += 1
        if self._chunk_step >= self._n_action_steps:
            self._chunk_step = 0  # Reset for next chunk

            if (
                state is not None
                and self._state_at_replan is not None
                and self._phase < self.NUM_PHASES
                and self._retry_count < self.max_retries
            ):
                movement = float(np.linalg.norm(state - self._state_at_replan))
                if movement < self.movement_threshold:
                    self._retry_count += 1
                    logger.warning(
                        f"[Recovery] Minimal movement detected over last chunk "
                        f"(L2={movement:.3f} < {self.movement_threshold}). "
                        f"Forcing re-plan (retry {self._retry_count}/{self.max_retries})."
                    )
                    self.policy.reset()
                    self._left_detector.reset()
                    self._right_detector.reset()
                    self._steps_in_phase = 0
                    self._state_at_replan = None

        self._step_total += 1

        # ----------------------------------------------------------------
        # 5. Delegate to base policy
        # ----------------------------------------------------------------
        return super().select_action(observation)
