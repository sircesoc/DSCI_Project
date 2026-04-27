"""DAgger (Dataset Aggregation) recording for FoldFlow.

Shared autonomy mode: the trained policy runs by default, and the human
operator can take over at any time using the SO-101 leader arms.

Controls:
    S key  — Start recording (same as normal recording)
    T key  — Take over: switch from policy to teleop control
    R key  — Release: switch from teleop back to policy control
    N key  — Save current episode as successful
    D key  — Discard current episode and re-record
    ESC    — Abort recording session

The entire episode (policy actions + human corrections) is recorded
as continuous training data. The policy learns both normal execution
and recovery behavior from these mixed episodes.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from pynput import keyboard as kb

from isaacsim.simulation_app import SimulationApp
from isaaclab.envs import DirectRLEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.devices import BiSO101Leader
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import (
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lehome.utils.logger import get_logger

from .common import stabilize_garment_after_reset
from .dataset_record import GARMENT_TYPE_TO_IDX

logger = get_logger(__name__)


class DAggerController:
    """Manages shared autonomy between policy and teleop."""

    def __init__(self, policy, teleop_interface, env):
        self.policy = policy
        self.teleop = teleop_interface
        self.env = env
        self.human_in_control = False
        self.flags = {
            "started": False,
            "success": False,
            "remove": False,
            "abort": False,
        }
        self._setup_keyboard()
        self._takeover_count = 0
        self._policy_steps = 0
        self._human_steps = 0

    def _setup_keyboard(self):
        """Set up keyboard listener for control switching."""
        def on_press(key):
            try:
                if key == kb.Key.esc:
                    self.flags["abort"] = True
                elif hasattr(key, 'char'):
                    if key.char == 's':
                        self.flags["started"] = True
                        logger.info("[DAgger] Recording started (S pressed)")
                    elif key.char == 't':
                        if not self.human_in_control:
                            self.human_in_control = True
                            self._takeover_count += 1
                            logger.info(f"[DAgger] TAKEOVER #{self._takeover_count} — human in control")
                    elif key.char == 'r':
                        if self.human_in_control:
                            self.human_in_control = False
                            self.policy.reset()  # clear action queue for fresh predictions
                            logger.info("[DAgger] RELEASE — policy in control")
                    elif key.char == 'n':
                        self.flags["success"] = True
                        logger.info("[DAgger] Episode marked as successful (N pressed)")
                    elif key.char == 'd':
                        self.flags["remove"] = True
                        logger.info("[DAgger] Episode discarded (D pressed)")
            except AttributeError:
                pass

        self.listener = kb.Listener(on_press=on_press)
        self.listener.daemon = True
        self.listener.start()

    def get_action(self, observation_dict: Dict[str, Any]) -> torch.Tensor:
        """Get action from either policy or teleop based on current mode.

        Returns:
            torch.Tensor action to apply to the environment.
        """
        if self.human_in_control:
            # Get action from SO-101 leader arms
            self._human_steps += 1
            try:
                dynamic_reset_gripper_effort_limit_sim(self.env, "bi-so101leader")
                action = self.teleop.advance()
                if action is None:
                    # Teleop not ready, maintain current position
                    return self._maintain_action(observation_dict)
                if isinstance(action, dict):
                    # Reset signal from teleop
                    if action.get("reset", False):
                        return self._maintain_action(observation_dict)
                return action
            except Exception as e:
                logger.error(f"[DAgger] Teleop error: {e}")
                return self._maintain_action(observation_dict)
        else:
            # Get action from trained policy
            self._policy_steps += 1
            action_np = self.policy.select_action(observation_dict)
            action = torch.from_numpy(action_np).float().to(self.env.device).unsqueeze(0)
            return action

    def _maintain_action(self, observation_dict: Dict[str, Any]) -> torch.Tensor:
        """Return current joint positions as action (hold still)."""
        state = observation_dict.get("observation.state", None)
        if state is not None:
            if isinstance(state, np.ndarray):
                return torch.from_numpy(state).float().unsqueeze(0).to(self.env.device)
        return torch.zeros(1, 12, dtype=torch.float32, device=self.env.device)

    def get_stats(self) -> str:
        return (f"policy_steps={self._policy_steps}, human_steps={self._human_steps}, "
                f"takeovers={self._takeover_count}")


def run_dagger_recording(
    env: DirectRLEnv,
    teleop_interface: BiSO101Leader,
    policy: Any,
    args: argparse.Namespace,
    dataset: LeRobotDataset,
    json_path: Path,
) -> None:
    """Run DAgger recording session.

    Policy runs by default. Human takes over with T key, releases with R key.
    Everything is recorded as training data.

    Args:
        env: Isaac Sim environment.
        teleop_interface: BiSO101Leader for human takeover.
        policy: Trained FoldFlow policy for autonomous execution.
        args: Command-line arguments.
        dataset: LeRobotDataset for saving episodes.
        json_path: Path for episode metadata.
    """
    controller = DAggerController(policy, teleop_interface, env)

    # Wait for S key to start
    logger.info("[DAgger] Press S to start recording")
    logger.info("[DAgger] During recording: T=takeover, R=release, N=save, D=discard, ESC=abort")

    while not controller.flags["started"]:
        if controller.flags["abort"]:
            logger.info("[DAgger] Aborted before starting")
            return
        # Keep sim alive
        env.render()
        time.sleep(0.01)

    # Initialize
    try:
        env.reset()
        stabilize_garment_after_reset(env, args)
        object_initial_pose = env.get_all_pose()
    except Exception as e:
        logger.error(f"[DAgger] Error during initial reset: {e}")
        import traceback
        traceback.print_exc()
        return
    policy.reset()

    episode_index = 0
    garment_type_str = getattr(env, "garment_type", None)
    if garment_type_str is None and hasattr(env, "cfg"):
        garment_type_str = getattr(env.cfg, "garment_type", "top-long-sleeve")

    while episode_index < args.num_episode:
        if controller.flags["abort"]:
            dataset.clear_episode_buffer()
            dataset.finalize()
            logger.warning(f"[DAgger] Aborted after {episode_index} episodes")
            return

        controller.flags["success"] = False
        controller.flags["remove"] = False
        controller._takeover_count = 0
        controller._policy_steps = 0
        controller._human_steps = 0

        logger.info(f"[DAgger] Episode {episode_index}/{args.num_episode} — policy in control")

        step = 0
        while not controller.flags["success"]:
            if controller.flags["abort"]:
                dataset.clear_episode_buffer()
                dataset.finalize()
                return

            # Get observation
            observation_dict = env._get_observations()
            observation_dict["observation.garment_type"] = np.array(
                GARMENT_TYPE_TO_IDX.get(garment_type_str or "top-long-sleeve", 0),
                dtype=np.float32,
            )

            # Get action (from policy or human)
            action = controller.get_action(observation_dict)

            # Step environment
            env.step(action)

            # Get post-step observations for recording
            observations = env._get_observations()
            if getattr(args, "disable_depth", False) and "observation.top_depth" in observations:
                observations.pop("observation.top_depth")

            _, truncated = env._get_dones()
            frame = {
                **observations,
                "task": getattr(args, "task_description", "fold garment"),
            }

            # Add garment type
            type_idx = GARMENT_TYPE_TO_IDX.get(garment_type_str or "top-long-sleeve", 0)
            frame["observation.garment_type"] = np.array([type_idx], dtype=np.int32)

            dataset.add_frame(frame)
            step += 1

            # Log status periodically
            if step % 100 == 0:
                mode = "HUMAN" if controller.human_in_control else "POLICY"
                logger.info(f"[DAgger] Step {step}, mode={mode}, {controller.get_stats()}")

            # Handle episode end
            if truncated or controller.flags["remove"]:
                dataset.clear_episode_buffer()
                if controller.flags["remove"]:
                    logger.info(f"[DAgger] Episode {episode_index} discarded, re-recording")
                else:
                    logger.info(f"[DAgger] Episode {episode_index} truncated, re-recording")
                env.reset()
                stabilize_garment_after_reset(env, args)
                object_initial_pose = env.get_all_pose()
                policy.reset()
                controller.flags["remove"] = False
                controller.human_in_control = False
                break

        if controller.flags["success"]:
            # Save episode
            logger.info(f"[DAgger] Saving episode {episode_index} ({controller.get_stats()})")
            try:
                dataset.save_episode()
                append_episode_initial_pose(
                    json_path, episode_index, object_initial_pose,
                    garment_name=getattr(env.cfg, "garment_name", None),
                    scale=getattr(env.object, "init_scale", None) if hasattr(env, "object") else None,
                )
            except Exception as e:
                logger.error(f"[DAgger] Failed to save episode: {e}")
                traceback.print_exc()

            episode_index += 1
            logger.info(f"[DAgger] Progress: {episode_index}/{args.num_episode}")

            # Reset for next episode
            env.reset()
            stabilize_garment_after_reset(env, args)
            object_initial_pose = env.get_all_pose()
            policy.reset()
            controller.human_in_control = False

    dataset.clear_episode_buffer()
    dataset.finalize()
    logger.info(f"[DAgger] All {args.num_episode} episodes completed!")
