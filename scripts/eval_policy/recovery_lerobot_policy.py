"""
RecoveryLeRobotPolicy
=====================
Extends LeRobotPolicy with a 5-state drop-recovery machine for clothes folding.

State machine
-------------
NORMAL       → run FoldFlow, watch DropDetector
DROP_DETECTED→ localise garment, start MPPI pickup
PICKUP       → drain MPPI actions one per step
REENTRY      → SDEdit-corrected action chunk pushed to queue
FALLBACK     → full re-plan from scratch (localization failed)

Registration
------------
Registered as "lerobot_recovery" in the policy registry.
Use --policy_type lerobot_recovery when calling eval.py.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict

import numpy as np
import torch

from lehome.utils.logger import get_logger
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry
from .drop_recovery import DropDetector, KeypointGarmentLocalizer, MPPIPickupPrimitive

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# State enum
# --------------------------------------------------------------------------

class _State(Enum):
    NORMAL = auto()
    DROP_DETECTED = auto()
    PICKUP = auto()
    REENTRY = auto()
    FALLBACK = auto()


# --------------------------------------------------------------------------
# GripperCycleDetector (kept for phase tracking)
# --------------------------------------------------------------------------

class GripperCycleDetector:
    """Detects a full grasp-release cycle on a single gripper.

    A cycle is: value drops below `close_thresh` (grasp) then rises above
    `open_thresh` (release).
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

    @property
    def is_closed(self) -> bool:
        return self._was_closed


# --------------------------------------------------------------------------
# Recovery policy
# --------------------------------------------------------------------------

@PolicyRegistry.register("lerobot_recovery")
class RecoveryLeRobotPolicy(LeRobotPolicy):
    """LeRobotPolicy with keypoint-guided drop-recovery for clothes-folding.

    Phase model (top_long has 3 gripper-cycle phases):
        Phase 0 → first sleeve fold  (gripper cycle #1)
        Phase 1 → second sleeve fold (gripper cycle #2)
        Phase 2 → final body fold    (gripper cycle #3)

    Recovery pipeline:
        NORMAL → (drop detected) → DROP_DETECTED → PICKUP → REENTRY → NORMAL
        DROP_DETECTED → (localization failed) → FALLBACK → NORMAL

    Args:
        t_inj:              SDEdit noise injection level [0,1] (default 0.4).
        localize_retries:   Max frames to attempt localization in DROP_DETECTED.
        enable_recovery:    Set False to disable recovery (fallback to timeout logic).
        close_thresh / open_thresh / min_close_steps: Gripper cycle detection params.
        keypoint_head:      Optional pre-loaded GarmentKeypointHead. If None,
                            drop recovery is limited to FALLBACK (full re-plan).
        fk_solver:          Optional RobotKinematics for MPPI. If None, MPPI is
                            skipped and only SDEdit re-entry is used after manual
                            repositioning.
        table_depth_mm:     Estimated table depth; auto-set on first observation.
    """

    LEFT_GRIPPER_IDX = 5
    RIGHT_GRIPPER_IDX = 11
    NUM_PHASES = 3

    def __init__(
        self,
        *args,
        t_inj: float = 0.4,
        localize_retries: int = 3,
        enable_recovery: bool = True,
        close_thresh: float = 0.35,
        open_thresh: float = 0.60,
        min_close_steps: int = 3,
        keypoint_head=None,
        fk_solver=None,
        table_depth_mm: float = 800.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t_inj = t_inj
        self.localize_retries = localize_retries
        self.enable_recovery = enable_recovery

        # Gripper cycle detectors (one per arm) — for phase tracking
        self._left_detector = GripperCycleDetector(close_thresh, open_thresh, min_close_steps)
        self._right_detector = GripperCycleDetector(close_thresh, open_thresh, min_close_steps)

        # Episode-level state
        self._phase = 0
        self._step_total = 0
        self._garment_type: str = "top-long-sleeve"
        self._state_machine: _State = _State.NORMAL
        self._dropped_arm: str | None = None
        self._localize_frame: int = 0
        self._pickup_queue: list[np.ndarray] = []
        self._x_failed = None  # saved last_chunk before drop
        self._table_depth_set = False

        # Sub-components
        self._drop_detector = DropDetector(table_depth_mm=table_depth_mm)

        self._localizer: KeypointGarmentLocalizer | None = None
        if keypoint_head is not None:
            backbone = self._get_backbone()
            if backbone is not None:
                self._localizer = KeypointGarmentLocalizer(
                    keypoint_head=keypoint_head,
                    backbone=backbone,
                    table_depth_mm=table_depth_mm,
                    device=str(self.device),
                )
                logger.info("[Recovery] KeypointGarmentLocalizer initialized.")

        self._mppi: MPPIPickupPrimitive | None = None
        if fk_solver is not None:
            self._mppi = MPPIPickupPrimitive(fk_solver)
            logger.info("[Recovery] MPPIPickupPrimitive initialized.")

    def _get_backbone(self):
        """Extract the shared ResNet18 backbone from the loaded policy."""
        try:
            return self.policy.model.vision_encoder.backbone
        except AttributeError:
            logger.warning("[Recovery] Could not extract backbone from policy.")
            return None

    def reset(self):
        super().reset()
        self._left_detector.reset()
        self._right_detector.reset()
        self._phase = 0
        self._step_total = 0
        self._garment_type = "top-long-sleeve"
        self._state_machine = _State.NORMAL
        self._dropped_arm = None
        self._localize_frame = 0
        self._pickup_queue = []
        self._x_failed = None
        self._table_depth_set = False
        self._drop_detector.reset()
        logger.info("[Recovery] Episode reset — state=NORMAL")

    # ------------------------------------------------------------------
    # Intentional release detection (cycle-based, for DropDetector)
    # ------------------------------------------------------------------

    def _update_phase(self, state: np.ndarray) -> bool:
        """Update gripper cycle detectors and return True if a cycle completed."""
        left_cycle = self._left_detector.update(float(state[self.LEFT_GRIPPER_IDX]))
        right_cycle = self._right_detector.update(float(state[self.RIGHT_GRIPPER_IDX]))
        if left_cycle or right_cycle:
            side = "left" if left_cycle else "right"
            logger.info(
                f"[Recovery] {side} gripper cycle → phase {self._phase} complete "
                f"at step {self._step_total}"
            )
            self._phase = min(self._phase + 1, self.NUM_PHASES)
            return True
        return False

    @property
    def _intentional_release(self) -> bool:
        """True if a gripper is in the process of an intentional release cycle."""
        # We treat any gripper that has been detected as closed (grasping) as
        # potentially in an intentional release — the phase tracker handles the
        # cycle-completion signal; this flag suppresses spurious Signal A fires
        # when the gripper opens as part of an intentional release.
        return self._left_detector.is_closed or self._right_detector.is_closed

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _transition_drop_detected(self, observation: Dict):
        """Handle transition to DROP_DETECTED state."""
        self._state_machine = _State.DROP_DETECTED
        self._localize_frame = 0
        # Save last chunk for SDEdit
        try:
            self._x_failed = self.policy.policy.last_chunk
        except AttributeError:
            self._x_failed = None
            logger.warning("[Recovery] Could not access last_chunk — SDEdit will full re-plan.")
        logger.info(f"[Recovery] DROP_DETECTED: arm={self._dropped_arm}")

    def _try_localize(self, observation: Dict) -> np.ndarray | None:
        """Attempt to localize the garment grasp point."""
        if self._localizer is None:
            return None
        rgb = observation.get("observation.images.top_rgb")
        depth = observation.get("observation.top_depth")
        if rgb is None or depth is None:
            logger.warning("[Recovery] top_rgb or top_depth not in observation for localization.")
            return None
        garment_type = getattr(self, "_garment_type", "top-long-sleeve")
        return self._localizer.localize(rgb, depth, fold_phase=self._phase, garment_type=garment_type)

    def _start_pickup(self, cloth_pt: np.ndarray, state: np.ndarray):
        """Start MPPI pickup primitive."""
        if self._mppi is None:
            logger.warning("[Recovery] No MPPI solver — skipping PICKUP, going to REENTRY.")
            self._transition_reentry()
            return
        self._mppi.reset(state, cloth_pt, arm=self._dropped_arm or "left")
        self._state_machine = _State.PICKUP
        logger.info("[Recovery] PICKUP started.")

    def _transition_reentry(self):
        """Transition to REENTRY: SDEdit correct trajectory."""
        self._state_machine = _State.REENTRY

    def _transition_fallback(self):
        """Transition to FALLBACK: full re-plan."""
        self._state_machine = _State.FALLBACK
        logger.warning("[Recovery] FALLBACK: full re-plan from scratch.")

    # ------------------------------------------------------------------
    # Main select_action
    # ------------------------------------------------------------------

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        state = observation.get("observation.state")
        depth = observation.get("observation.top_depth")
        # Track garment type if provided (e.g. from env metadata or observation dict)
        if "observation.garment_type_str" in observation:
            self._garment_type = observation["observation.garment_type_str"]

        # ----------------------------------------------------------------
        # Initialise table depth on first step
        # ----------------------------------------------------------------
        if not self._table_depth_set and depth is not None:
            self._drop_detector.set_table_depth(depth)
            if self._localizer is not None:
                self._localizer.table_depth_mm = self._drop_detector.table_depth_mm
            self._table_depth_set = True

        # ----------------------------------------------------------------
        # Phase tracking (all states)
        # ----------------------------------------------------------------
        if state is not None:
            self._update_phase(state)

        # ================================================================
        # State machine
        # ================================================================

        if self._state_machine == _State.NORMAL:
            # Check for drops
            if self.enable_recovery and state is not None and depth is not None:
                drop, arm = self._drop_detector.update(
                    state, depth, intentional_release=self._intentional_release
                )
                if drop:
                    self._dropped_arm = arm
                    self._transition_drop_detected(observation)

            # Normal execution
            return super().select_action(observation)

        elif self._state_machine == _State.DROP_DETECTED:
            self._localize_frame += 1
            cloth_pt = self._try_localize(observation)
            if cloth_pt is not None:
                self._start_pickup(cloth_pt, state if state is not None else np.zeros(12))
                # Return current (held) action while transitioning
                return self._held_action(state)
            if self._localize_frame >= self.localize_retries:
                self._transition_fallback()
                return self._do_fallback()
            # Waiting for localization — hold current position
            return self._held_action(state)

        elif self._state_machine == _State.PICKUP:
            if state is None:
                return self._held_action(state)
            action_12, done = self._mppi.step(state)
            if done:
                self._transition_reentry()
                logger.info("[Recovery] PICKUP complete → REENTRY")
            return action_12.astype(np.float32)

        elif self._state_machine == _State.REENTRY:
            return self._do_reentry(observation)

        elif self._state_machine == _State.FALLBACK:
            return self._do_fallback()

        # Should never reach here
        return super().select_action(observation)

    # ------------------------------------------------------------------
    # Sub-actions
    # ------------------------------------------------------------------

    def _held_action(self, state: np.ndarray | None) -> np.ndarray:
        """Return a zero-velocity action (hold current joint positions)."""
        if state is not None:
            return state.astype(np.float32)
        return np.zeros(12, dtype=np.float32)

    def _do_reentry(self, observation: Dict) -> np.ndarray:
        """Apply SDEdit and push corrected chunk, then return to NORMAL."""
        try:
            inner_policy = self.policy.policy  # FoldFlowPolicy
            # Flush stale obs queues and repopulate from current observation
            inner_policy.reset()
            # Preprocess and populate queues with current obs
            # (delegate to super so the queues are filled via the normal pipeline)
            # We call select_action twice to fill n_obs_steps queues, but first
            # just repopulate via a single forward pass without generating actions.
            # Simplest approach: call super once to refill queues, then apply SDEdit.
            _ = super().select_action(observation)

            if self._x_failed is not None:
                device = next(inner_policy.parameters()).device
                x_failed = self._x_failed.to(device)
                obs_cond = inner_policy.encode_obs_from_queues()
                corrected = inner_policy.model.correct_trajectory(
                    x_failed, t_inj=self.t_inj, obs_cond=obs_cond
                )
                # Push corrected chunk into the action queue
                from lerobot.utils.constants import ACTION
                inner_policy._queues[ACTION].clear()
                start = inner_policy.config.n_obs_steps - 1
                end = start + inner_policy.config.n_action_steps
                chunk_slice = corrected[0, start:end]  # (n_action_steps, action_dim)
                inner_policy._queues[ACTION].extend(chunk_slice.unbind(0))
                logger.info("[Recovery] SDEdit re-entry: corrected chunk pushed.")
                self._state_machine = _State.NORMAL
                from lerobot.utils.constants import ACTION as _ACT
                if len(inner_policy._queues[_ACT]) > 0:
                    action = inner_policy._queues[_ACT].popleft()
                    return action.cpu().numpy()
            else:
                # No saved chunk: full re-plan
                inner_policy.reset()
                self._state_machine = _State.NORMAL
        except Exception as e:
            logger.error(f"[Recovery] SDEdit re-entry failed: {e}; falling back to full re-plan.")
            self._transition_fallback()
            return self._do_fallback()

        self._state_machine = _State.NORMAL
        return super().select_action(observation)

    def _do_fallback(self) -> np.ndarray:
        """Full re-plan: reset inner policy and generate fresh chunk."""
        try:
            self.policy.reset()
        except Exception:
            pass
        self._state_machine = _State.NORMAL
        self._x_failed = None
        logger.info("[Recovery] FALLBACK complete → NORMAL (full re-plan).")
        return np.zeros(12, dtype=np.float32)  # one no-op step while queues refill
