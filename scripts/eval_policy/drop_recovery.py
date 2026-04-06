"""Drop recovery components for FoldFlow: detector, localizer, MPPI pickup primitive.

Components
----------
DropDetector           — detects unexpected gripper opens + cloth-on-table depth anomaly
KeypointGarmentLocalizer — backprojects best grasp keypoint to 3D world coords
MPPIPickupPrimitive    — FK-based MPPI controller: REACH → GRASP → LIFT
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from lehome.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DropDetector
# ---------------------------------------------------------------------------

class DropDetector:
    """Detects accidental garment drops using gripper state + depth anomaly.

    Signal A (state): unexpected gripper open during a non-release phase.
    Signal B (vision): cloth-on-table depth score jump in workspace ROI.

    Combined logic: Signal A triggers a candidate drop; Signal B confirms
    within `confirm_frames` steps; unconfirmed candidates are cleared.

    Args:
        table_depth_mm:  Estimated table surface depth in millimetres (set at
                         episode start via `set_table_depth`).
        gripper_close:   Gripper value below which the gripper is considered closed.
        gripper_open:    Gripper value above which the gripper is considered open.
        cloth_score_jump: Minimum change in cloth-on-table score to confirm a drop.
        confirm_frames:  Max frames between Signal A and Signal B to confirm drop.
        roi_size:        Side length (pixels) of the square workspace ROI.
    """

    LEFT_GRIPPER_IDX = 5
    RIGHT_GRIPPER_IDX = 11

    def __init__(
        self,
        table_depth_mm: float = 800.0,
        gripper_close: float = 0.35,
        gripper_open: float = 0.60,
        cloth_score_jump: float = 0.15,
        confirm_frames: int = 8,
        roi_size: int = 200,
    ):
        self.table_depth_mm = table_depth_mm
        self.gripper_close = gripper_close
        self.gripper_open = gripper_open
        self.cloth_score_jump = cloth_score_jump
        self.confirm_frames = confirm_frames
        self.roi_size = roi_size

        self._prev_gripper = {
            "left": 1.0,
            "right": 1.0,
        }
        self._prev_gripper_was_closed = {"left": False, "right": False}
        self._candidate_frames = 0   # frames since Signal A fired
        self._candidate_arm: str | None = None
        self._prev_cloth_score: float = 0.0

    def set_table_depth(self, depth_image: np.ndarray) -> None:
        """Estimate table depth from first-frame depth median. Call at episode start."""
        self.table_depth_mm = float(np.median(depth_image[depth_image > 0]))
        self._prev_cloth_score = self._cloth_score(depth_image)
        logger.debug(f"[DropDetector] table_depth_mm={self.table_depth_mm:.1f}")

    def reset(self) -> None:
        self._prev_gripper = {"left": 1.0, "right": 1.0}
        self._prev_gripper_was_closed = {"left": False, "right": False}
        self._candidate_frames = 0
        self._candidate_arm = None
        self._prev_cloth_score = 0.0

    def _cloth_score(self, depth: np.ndarray) -> float:
        """Fraction of workspace ROI pixels at cloth-on-table depth."""
        h, w = depth.shape
        cy, cx = h // 2, w // 2
        r = self.roi_size // 2
        roi = depth[cy - r:cy + r, cx - r:cx + r]
        lo = self.table_depth_mm - 40.0
        hi = self.table_depth_mm - 5.0
        mask = (roi > lo) & (roi < hi) & (roi > 0)
        return float(mask.sum()) / max(1, mask.size)

    def update(
        self,
        state: np.ndarray,
        depth_image: np.ndarray,
        intentional_release: bool,
    ) -> tuple[bool, str | None]:
        """Feed one timestep of state + depth.

        Returns:
            (drop_detected, arm) — arm is "left" or "right" (or None if no drop).
        """
        left_val = float(state[self.LEFT_GRIPPER_IDX])
        right_val = float(state[self.RIGHT_GRIPPER_IDX])

        # Track closed → open transition (unexpected open = potential drop)
        dropped_arm: str | None = None
        if not intentional_release:
            for arm, val, prev in [
                ("left", left_val, self._prev_gripper["left"]),
                ("right", right_val, self._prev_gripper["right"]),
            ]:
                was_closed = self._prev_gripper_was_closed[arm]
                if was_closed and val > self.gripper_open:
                    dropped_arm = arm
                    logger.debug(f"[DropDetector] Signal A: {arm} gripper unexpected open")
                # Update closed state
                if val < self.gripper_close:
                    self._prev_gripper_was_closed[arm] = True
                elif val > self.gripper_open:
                    self._prev_gripper_was_closed[arm] = False

        self._prev_gripper["left"] = left_val
        self._prev_gripper["right"] = right_val

        # Signal A: set candidate
        if dropped_arm is not None:
            self._candidate_arm = dropped_arm
            self._candidate_frames = 0

        # Signal B: depth anomaly
        cloth_score = self._cloth_score(depth_image)
        score_jump = cloth_score - self._prev_cloth_score
        self._prev_cloth_score = cloth_score

        if self._candidate_arm is not None:
            self._candidate_frames += 1
            if score_jump > self.cloth_score_jump:
                arm = self._candidate_arm
                self._candidate_arm = None
                self._candidate_frames = 0
                logger.info(f"[DropDetector] Drop confirmed: {arm} arm, cloth_score_jump={score_jump:.3f}")
                return True, arm
            if self._candidate_frames >= self.confirm_frames:
                # Candidate timed out without depth confirmation — clear
                logger.debug("[DropDetector] Candidate timed out without depth confirmation")
                self._candidate_arm = None
                self._candidate_frames = 0

        return False, None


# ---------------------------------------------------------------------------
# KeypointGarmentLocalizer
# ---------------------------------------------------------------------------

class KeypointGarmentLocalizer:
    """Localizes garment grasp point using the trained GarmentKeypointHead.

    Pipeline:
        top_rgb → backbone (shared with main encoder) → keypoint_head
        → best keypoint uv → backproject with depth → world xyz

    Phase-to-keypoint mapping:
        Phase 0: kp[0] (left sleeve tip)
        Phase 1: kp[1] (right sleeve tip)
        Phase 2: kp[2] or kp[3] (shoulder region)

    Falls back to depth centroid if keypoint confidence is below threshold.

    Args:
        keypoint_head:      Trained GarmentKeypointHead.
        backbone:           ResNet18 backbone (no FC) — shared with main encoder.
        table_depth_mm:     Table surface depth for fallback centroid masking.
        confidence_thresh:  Minimum soft-argmax weight mass to accept a keypoint.
        device:             Torch device.
    """

    # Camera intrinsics (top camera)
    FX = FY = 482.0
    CX, CY = 320.0, 240.0
    IMG_W, IMG_H = 640, 480

    def __init__(
        self,
        keypoint_head: nn.Module,
        backbone: nn.Module,
        table_depth_mm: float = 800.0,
        confidence_thresh: float = 0.5,
        device: str = "cuda",
    ):
        self.keypoint_head = keypoint_head
        self.backbone = backbone
        self.table_depth_mm = table_depth_mm
        self.confidence_thresh = confidence_thresh
        self.device = torch.device(device)

        # Pre-compute transform matrices
        import scipy.spatial.transform as sci_tf
        R_usd = sci_tf.Rotation.from_quat([-0.9862856, 0, 0, 0.1650476]).as_matrix().astype(np.float32)
        R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        self._R_mix = (R_usd @ R_opt).astype(np.float32)  # cam_to_robot rotation
        self._t_cam2robot = np.array([0.225, -0.5, 0.6], dtype=np.float32)
        self._R_robot2world = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        self._t_robot2world = np.array([0.23, -0.25, 0.5], dtype=np.float32)

    # Keypoint indices to target per fold phase, per garment type.
    _PHASE_TO_KP: dict[str, dict[int, list[int]]] = {
        "top-long-sleeve":  {0: [2], 1: [3], 2: [0, 1]},
        "top-short-sleeve": {0: [2], 1: [3], 2: [0, 1]},
        "long-pant":        {0: [2], 1: [3], 2: [0, 1]},
        "short-pant":       {0: [4], 1: [5], 2: [0, 1]},
    }

    def _phase_to_kp_idx(self, fold_phase: int, garment_type: str = "top-long-sleeve") -> list[int]:
        """Return candidate keypoint indices for a given fold phase and garment type."""
        type_map = self._PHASE_TO_KP.get(garment_type, self._PHASE_TO_KP["top-long-sleeve"])
        return type_map.get(fold_phase, [0])

    def _backproject(self, u: float, v: float, depth_mm: float) -> np.ndarray:
        """Backproject normalized pixel (u,v) + depth to world xyz."""
        px = u * self.IMG_W
        py = v * self.IMG_H
        z = depth_mm / 1000.0  # mm → m
        x_cam = (px - self.CX) * z / self.FX
        y_cam = (py - self.CY) * z / self.FY
        cam_pt = np.array([x_cam, y_cam, z], dtype=np.float32)

        # Camera → RobotBase
        robot_pt = self._R_mix @ cam_pt + self._t_cam2robot

        # RobotBase → World (inverse: world_pt = inv(R_w2r) @ (robot_pt - t_robot))
        # R_w2r = [[-1,0,0],[0,-1,0],[0,0,1]], t = [0.23,-0.25,0.5]
        # world_pt = R_robot2world @ robot_pt + t_robot2world
        # Actually: R_w2r maps world→robot, so robot = R_w2r @ (world - t)
        # world = R_w2r^T @ robot + t (since R_w2r is orthogonal)
        world_pt = self._R_robot2world.T @ robot_pt + self._t_robot2world
        return world_pt

    @torch.no_grad()
    def localize(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        fold_phase: int = 0,
        garment_type: str = "top-long-sleeve",
    ) -> np.ndarray | None:
        """Localize garment grasp point from top-camera observation.

        Args:
            rgb:          (H, W, 3) uint8 top-camera RGB frame.
            depth:        (H, W) uint16 depth in mm.
            fold_phase:   Current fold phase (0-2).
            garment_type: Garment type string for phase-to-keypoint mapping.

        Returns:
            (3,) world xyz in metres, or None if localization failed.
        """
        # Preprocess RGB: (H, W, 3) → (1, 3, H, W) float [0,1]
        img = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img = img.to(self.device)

        # Backbone features (no crop needed here — use center crop for consistency)
        import torchvision
        crop = torchvision.transforms.CenterCrop((224, 224))
        img_crop = crop(img)
        features = self.backbone(img_crop)  # (1, C, H', W')

        coords, heatmaps = self.keypoint_head(features)  # (1, 6, 2), (1, 6, H', W')
        coords = coords[0].cpu().numpy()  # (6, 2)

        # Confidence: max heatmap value (after softmax) as proxy
        flat = heatmaps[0]  # (6, H', W')
        B, H, W = flat.shape
        conf = flat.reshape(B, -1).softmax(dim=-1).max(dim=-1).values.cpu().numpy()  # (6,)

        candidate_indices = self._phase_to_kp_idx(fold_phase, garment_type)
        best_idx = max(candidate_indices, key=lambda i: conf[i])

        if conf[best_idx] < self.confidence_thresh:
            logger.warning(
                f"[Localizer] Keypoint confidence {conf[best_idx]:.3f} < threshold "
                f"{self.confidence_thresh}; falling back to depth centroid."
            )
            return self._depth_centroid_fallback(depth)

        u, v = float(coords[best_idx, 0]), float(coords[best_idx, 1])
        px = int(round(u * self.IMG_W))
        py = int(round(v * self.IMG_H))
        px = np.clip(px, 0, self.IMG_W - 1)
        py = np.clip(py, 0, self.IMG_H - 1)
        depth_mm = float(depth[py, px])

        if depth_mm <= 0:
            logger.warning("[Localizer] Zero depth at keypoint pixel; falling back to depth centroid.")
            return self._depth_centroid_fallback(depth)

        world_xyz = self._backproject(u, v, depth_mm)
        logger.debug(f"[Localizer] Keypoint {best_idx} uv=({u:.3f},{v:.3f}) → world={world_xyz}")
        return world_xyz

    def _depth_centroid_fallback(self, depth: np.ndarray) -> np.ndarray | None:
        """Find cloth centroid using depth thresholding."""
        lo = self.table_depth_mm - 40.0
        hi = self.table_depth_mm - 5.0
        mask = (depth > lo) & (depth < hi) & (depth > 0)
        if mask.sum() < 10:
            logger.warning("[Localizer] Depth fallback: insufficient cloth pixels found.")
            return None
        ys, xs = np.where(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        depth_mm = float(depth[int(cy), int(cx)])
        return self._backproject(cx / self.IMG_W, cy / self.IMG_H, depth_mm)


# ---------------------------------------------------------------------------
# MPPIPickupPrimitive
# ---------------------------------------------------------------------------

_HANDOFF_WORLD_XYZ = np.array([0.0, 0.0, 0.35], dtype=np.float32)  # safe lift pose


class MPPIPickupPrimitive:
    """FK-based MPPI controller for pickup: REACH → GRASP → LIFT.

    Samples K random joint-delta trajectories, rolls them out under a simple
    integrator, evaluates FK-based cost, returns weighted-average optimal action.

    Args:
        fk_solver:    RobotKinematics instance (takes degrees, returns 4×4).
        n_samples:    Number of MPPI trajectory samples (K).
        horizon:      Planning horizon (H) in steps.
        noise_sigma:  Std of joint-delta noise (radians).
        temperature:  MPPI temperature λ.
        q_min:        (6,) lower joint limits in radians.
        q_max:        (6,) upper joint limits in radians.
    """

    # Phase identifiers
    REACH = "REACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    DONE = "DONE"

    REACH_THRESH = 0.03   # m
    LIFT_THRESH = 0.05    # m
    GRASP_FRAMES = 10     # frames to ramp gripper closed

    def __init__(
        self,
        fk_solver,
        n_samples: int = 200,
        horizon: int = 30,
        noise_sigma: float = 0.05,
        temperature: float = 0.01,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
    ):
        self.fk = fk_solver
        self.K = n_samples
        self.H = horizon
        self.sigma = noise_sigma
        self.lam = temperature
        self.q_min = q_min if q_min is not None else np.full(6, -np.pi)
        self.q_max = q_max if q_max is not None else np.full(6, np.pi)

        self._phase: str = self.DONE
        self._cloth_target: np.ndarray | None = None
        self._arm: str = "left"
        self._arm_idx_slice: slice = slice(0, 6)   # which half of 12-dim state
        self._held_pos: np.ndarray = np.zeros(6)   # idle arm position
        self._U: np.ndarray = np.zeros((horizon, 6))  # warm-start action sequence
        self._grasp_frame: int = 0
        self._current_q: np.ndarray = np.zeros(6)

        # Cost weights
        self._w_dist = 1.0
        self._w_smooth = 0.1
        self._w_terminal = 5.0

    def reset(
        self,
        current_state: np.ndarray,
        cloth_world_point: np.ndarray,
        arm: str,
    ) -> None:
        """Initialise pickup from current robot state.

        Args:
            current_state:    (12,) full bimanual robot state in radians.
            cloth_world_point: (3,) world xyz of the grasp target.
            arm:              "left" or "right" — which arm to control.
        """
        self._arm = arm
        self._arm_idx_slice = slice(0, 6) if arm == "left" else slice(6, 12)
        idle_slice = slice(6, 12) if arm == "left" else slice(0, 6)
        self._current_q = current_state[self._arm_idx_slice].copy()
        self._held_pos = current_state[idle_slice].copy()
        self._cloth_target = cloth_world_point.copy()
        self._U = np.zeros((self.H, 6))
        self._phase = self.REACH
        self._grasp_frame = 0
        logger.info(
            f"[MPPI] Reset: arm={arm}, target={cloth_world_point}, phase=REACH"
        )

    def _fk_xyz(self, q_rad: np.ndarray) -> np.ndarray:
        """Forward kinematics: (6,) radians → (3,) world xyz."""
        T = self.fk.forward_kinematics(np.degrees(q_rad))
        return T[:3, 3].astype(np.float32)

    def _mppi_step(self, target_xyz: np.ndarray) -> np.ndarray:
        """Run one MPPI optimisation step toward target_xyz.

        Returns:
            (6,) next joint position for the active arm (radians).
        """
        rng = np.random.default_rng()
        eps = rng.standard_normal((self.K, self.H, 6)) * self.sigma  # (K, H, 6)
        perturbed_U = self._U[np.newaxis] + eps  # (K, H, 6)

        costs = np.zeros(self.K, dtype=np.float64)
        for k in range(self.K):
            q = self._current_q.copy()
            for h in range(self.H):
                q_next = np.clip(q + perturbed_U[k, h], self.q_min, self.q_max)
                xyz = self._fk_xyz(q_next)
                dist_cost = np.sum((xyz - target_xyz) ** 2)
                smooth_cost = np.sum(perturbed_U[k, h] ** 2)
                costs[k] += self._w_dist * dist_cost + self._w_smooth * smooth_cost
                q = q_next
            # Terminal cost
            costs[k] += self._w_terminal * np.sum((self._fk_xyz(q) - target_xyz) ** 2)

        beta = costs.min()
        w = np.exp(-(costs - beta) / self.lam)
        w /= w.sum() + 1e-8
        self._U += np.einsum("k,khj->hj", w, eps)  # (H, 6) weighted update

        next_q = np.clip(self._current_q + self._U[0], self.q_min, self.q_max)
        # Receding horizon shift
        self._U = np.roll(self._U, -1, axis=0)
        self._U[-1] = 0.0
        return next_q

    def step(self, current_state: np.ndarray) -> tuple[np.ndarray, bool]:
        """Execute one control step.

        Args:
            current_state: (12,) full bimanual state in radians.

        Returns:
            (action_12dim, done) — action in radians, done=True when LIFT complete.
        """
        self._current_q = current_state[self._arm_idx_slice].copy()
        current_xyz = self._fk_xyz(self._current_q)

        action_arm = self._current_q.copy()
        action_gripper_val = 0.8  # open by default

        if self._phase == self.REACH:
            target = self._cloth_target.copy()
            target[2] += 0.02  # approach slightly above cloth
            action_arm = self._mppi_step(target)
            dist = float(np.linalg.norm(current_xyz - target))
            if dist < self.REACH_THRESH:
                self._phase = self.GRASP
                self._grasp_frame = 0
                logger.info("[MPPI] Phase: REACH → GRASP")

        elif self._phase == self.GRASP:
            # Hold position, ramp gripper closed
            action_arm = self._current_q.copy()
            progress = self._grasp_frame / max(1, self.GRASP_FRAMES)
            action_gripper_val = max(0.0, 0.8 - progress * 0.8)  # ramp 0.8 → 0.0
            self._grasp_frame += 1
            gripper_actual = current_state[self._arm_idx_slice.stop - 1]
            if self._grasp_frame >= self.GRASP_FRAMES or gripper_actual < 0.3:
                self._phase = self.LIFT
                logger.info("[MPPI] Phase: GRASP → LIFT")

        elif self._phase == self.LIFT:
            action_arm = self._mppi_step(_HANDOFF_WORLD_XYZ)
            action_gripper_val = 0.0  # keep closed
            dist = float(np.linalg.norm(current_xyz - _HANDOFF_WORLD_XYZ))
            if dist < self.LIFT_THRESH:
                self._phase = self.DONE
                logger.info("[MPPI] Phase: LIFT → DONE")

        elif self._phase == self.DONE:
            # Return current position (no-op)
            pass

        # Build 6-dim arm action: last element is gripper
        full_arm_action = action_arm.copy()
        full_arm_action[-1] = action_gripper_val

        # Assemble full 12-dim bimanual action
        action_12 = np.zeros(12, dtype=np.float32)
        action_12[self._arm_idx_slice] = full_arm_action
        idle_slice = slice(6, 12) if self._arm == "left" else slice(0, 6)
        action_12[idle_slice] = self._held_pos

        done = self._phase == self.DONE
        return action_12, done
