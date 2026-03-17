"""
Training module for RL-based reading policy in LatentMAS.

This module provides:
- Transition: Data structure for storing experience
- TrajectoryBuffer: Buffer for collecting trajectories
- RewardCalculator: Compute multi-component rewards
- GRPOTrainer: Group Relative Policy Optimization trainer
"""

from .trajectory import Transition, TrajectoryBuffer
from .reward import RewardCalculator
from .rl_trainer import GRPOTrainer

__all__ = [
    "Transition",
    "TrajectoryBuffer",
    "RewardCalculator",
    "GRPOTrainer",
]
