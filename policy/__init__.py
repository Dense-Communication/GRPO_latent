"""
Policy module for RL-based reading policy in LatentMAS.

This module provides:
- ReadingPolicyNetwork: Cross-attention based policy for block selection
- BlockSelector: Utility for top-k block selection with sampling
"""

from .reading_policy import ReadingPolicyNetwork
from .block_selector import BlockSelector

__all__ = [
    "ReadingPolicyNetwork",
    "BlockSelector",
]
