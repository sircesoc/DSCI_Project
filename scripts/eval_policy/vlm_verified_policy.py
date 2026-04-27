"""VLM-Verified Temporal Ensembling policy for FoldFlow.

Uses SmolVLM-256M as a real-time grasp verifier at chunk boundaries.
When the gripper is closed, queries the VLM with the wrist camera image:
"Is the robot gripper holding fabric?"

If the VLM says "no" → grasp failed → re-open gripper briefly, then re-plan.
If the VLM says "yes" → continue executing normally.

This provides semantic closed-loop control without changing the base policy.
"""

import math
import threading
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image

from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.policies.utils import populate_queues

from lehome.utils.logger import get_logger
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


class GraspVerifier:
    """Lightweight VLM-based grasp verification using SmolVLM-256M."""

    def __init__(self, device: str = "cpu"):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
        logger.info(f"[GraspVerifier] Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.device = device
        logger.info("[GraspVerifier] Ready")

    @torch.no_grad()
    def is_holding_fabric(self, wrist_image: np.ndarray) -> bool:
        """Query VLM: is the gripper holding fabric?

        Args:
            wrist_image: (H, W, 3) uint8 numpy array from wrist camera.

        Returns:
            True if VLM thinks gripper is holding fabric.
        """
        pil_image = Image.fromarray(wrist_image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Is the robot gripper holding fabric or cloth? Answer only yes or no."},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt, images=[pil_image], return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=5)
        response = self.processor.decode(output[0], skip_special_tokens=True).strip().lower()

        # Extract yes/no from response
        holding = "yes" in response
        logger.debug(f"[GraspVerifier] Response: '{response}' → holding={holding}")
        return holding


@PolicyRegistry.register("lerobot_vlm")
class VLMVerifiedLeRobotPolicy(LeRobotPolicy):
    """FoldFlow + Temporal Ensembling + VLM grasp verification.

    At each chunk boundary (every n_action_steps), if the gripper is closed,
    asks the VLM whether the gripper is actually holding fabric. If not,
    forces a re-plan to attempt recovery.
    """

    def __init__(self, *args, ensemble_decay: float = 0.1, replan_every: int = 4,
                 gripper_close_threshold: float = -0.05,
                 max_replan_attempts: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_decay = ensemble_decay
        self.replan_every = replan_every
        self.gripper_close_threshold = gripper_close_threshold
        self.max_replan_attempts = max_replan_attempts

        # State
        self._chunk_buffer: list = []
        self._step = 0
        self._replan_attempts = 0
        self._last_verification_step = -100
        self._grasp_failed = False  # async result from VLM
        self._vlm_busy = False

        # Initialize VLM verifier on CPU (GPU is full with sim)
        self._verifier = GraspVerifier(device="cpu")

        logger.info(f"[VLM] VLM-verified TE policy: replan_every={replan_every}, "
                    f"gripper_thresh={gripper_close_threshold}")

    def reset(self):
        super().reset()
        self._chunk_buffer = []
        self._step = 0
        self._replan_attempts = 0
        self._last_verification_step = -100

    def _is_gripper_closed(self, observation: Dict[str, Any]) -> bool:
        """Check if either gripper is in closed state from robot state."""
        state = observation.get("observation.state", None)
        if state is None:
            return False
        # Left gripper = index 5, Right gripper = index 11
        if isinstance(state, np.ndarray):
            left_grip = state[5] if len(state) > 5 else 0
            right_grip = state[11] if len(state) > 11 else 0
        else:
            left_grip = state[5].item() if state.shape[-1] > 5 else 0
            right_grip = state[11].item() if state.shape[-1] > 11 else 0

        # Log gripper values periodically for calibration
        if self._step % 50 == 0:
            print(f"[VLM] Step {self._step}: left_grip={left_grip:.3f}, right_grip={right_grip:.3f}", flush=True)

        return left_grip < self.gripper_close_threshold or right_grip < self.gripper_close_threshold

    def _get_wrist_image(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get the wrist camera image for grasp verification."""
        # Try left wrist first, then right
        for key in ["observation.images.left_rgb", "observation.images.right_rgb"]:
            if key in observation:
                img = observation[key]
                if isinstance(img, np.ndarray):
                    return img
                elif isinstance(img, torch.Tensor):
                    return img.cpu().numpy()
        return None

    def _vlm_check_thread(self, wrist_img: np.ndarray):
        """Run VLM inference in background thread."""
        holding = self._verifier.is_holding_fabric(wrist_img)
        if not holding:
            self._grasp_failed = True
            self._replan_attempts += 1
            print(f"[VLM] Grasp verification FAILED at step {self._step} "
                  f"(attempt {self._replan_attempts}/{self.max_replan_attempts})", flush=True)
        else:
            self._grasp_failed = False
            self._replan_attempts = 0
            print(f"[VLM] Grasp verification PASSED at step {self._step}", flush=True)
        self._vlm_busy = False

    def _verify_grasp_async(self, observation: Dict[str, Any]):
        """Launch async VLM grasp verification if gripper is closed."""
        if self._vlm_busy:
            return  # already checking

        if not self._is_gripper_closed(observation):
            return  # gripper open, no verification needed

        # Don't verify too frequently (minimum 16 steps between checks)
        if self._step - self._last_verification_step < 16:
            return

        wrist_img = self._get_wrist_image(observation)
        if wrist_img is None:
            return

        # Ensure image is uint8 HWC format
        if wrist_img.dtype != np.uint8:
            if wrist_img.max() <= 1.0:
                wrist_img = (wrist_img * 255).astype(np.uint8)
            else:
                wrist_img = wrist_img.astype(np.uint8)

        if wrist_img.ndim == 3 and wrist_img.shape[0] == 3:
            wrist_img = wrist_img.transpose(1, 2, 0)

        self._last_verification_step = self._step
        self._vlm_busy = True
        thread = threading.Thread(target=self._vlm_check_thread, args=(wrist_img.copy(),))
        thread.daemon = True
        thread.start()

    def select_action(self, observation: Dict[str, Any]):
        fp = self.policy

        # Store raw observation for VLM verification before preprocessing
        raw_observation = observation

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

        # Launch async VLM verification every 16 steps
        if self._step > 0 and self._step % 16 == 0:
            self._verify_grasp_async(raw_observation)

        # Check if previous async verification detected a failed grasp
        if self._grasp_failed and self._replan_attempts <= self.max_replan_attempts:
            self._chunk_buffer = []  # force replan
            self._grasp_failed = False
            print(f"[VLM] Forcing replan due to failed grasp", flush=True)

        # Regenerate chunk every replan_every steps
        if self._step % self.replan_every == 0 or len(self._chunk_buffer) == 0:
            with torch.inference_mode():
                chunk = fp.predict_action_chunk(batch_obs)
            self._chunk_buffer.append(chunk)

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
