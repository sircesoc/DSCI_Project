"""
LeHome Challenge Policy Module

This module provides the base policy interface and implementations
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
from .lerobot_policy import LeRobotPolicy
from .recovery_lerobot_policy import RecoveryLeRobotPolicy
from .phase_gated_policy import PhaseGatedLeRobotPolicy
from .temporal_ensembling_policy import TemporalEnsemblingLeRobotPolicy
from .example_participant_policy import CustomPolicy

__all__ = [
    "BasePolicy",
    "PolicyRegistry",
    "LeRobotPolicy",
    "RecoveryLeRobotPolicy",
    "PhaseGatedLeRobotPolicy",
    "TemporalEnsemblingLeRobotPolicy",
    "CustomPolicy",
]
