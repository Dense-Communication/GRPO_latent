"""
Trajectory data structures for RL training in LatentMAS.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import random
from collections import deque

import torch


@dataclass
class Transition:
    """
    A single transition in the RL trajectory.

    Represents one decision point where the policy selected blocks
    and received a reward.
    """
    # State information
    query_embed: torch.Tensor          # [D] query embedding used for selection
    block_summaries: torch.Tensor      # [num_blocks, D] available block summaries

    # Action information
    selected_indices: torch.Tensor     # [k] indices of selected blocks
    log_probs: torch.Tensor            # [k] log probabilities of selections
    probs: Optional[torch.Tensor] = None  # [num_blocks] full probability distribution

    # Reward information
    reward: float = 0.0
    task_reward: float = 0.0           # R1: task correctness
    consistency_reward: float = 0.0     # R2: evidence consistency
    cost_penalty: float = 0.0          # R3: read cost

    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> 'Transition':
        """Move all tensors to specified device."""
        return Transition(
            query_embed=self.query_embed.to(device),
            block_summaries=self.block_summaries.to(device),
            selected_indices=self.selected_indices.to(device),
            log_probs=self.log_probs.to(device),
            probs=self.probs.to(device) if self.probs is not None else None,
            reward=self.reward,
            task_reward=self.task_reward,
            consistency_reward=self.consistency_reward,
            cost_penalty=self.cost_penalty,
            metadata=self.metadata,
        )

    def detach(self) -> 'Transition':
        """Detach all tensors from computation graph."""
        return Transition(
            query_embed=self.query_embed.detach(),
            block_summaries=self.block_summaries.detach(),
            selected_indices=self.selected_indices.detach(),
            log_probs=self.log_probs.detach(),
            probs=self.probs.detach() if self.probs is not None else None,
            reward=self.reward,
            task_reward=self.task_reward,
            consistency_reward=self.consistency_reward,
            cost_penalty=self.cost_penalty,
            metadata=self.metadata,
        )


class TrajectoryBuffer:
    """
    Buffer for storing and sampling RL trajectories.

    Supports:
    - Adding individual transitions
    - Grouping transitions (for GRPO)
    - Random sampling
    - Prioritized sampling based on reward
    """

    def __init__(
        self,
        max_size: int = 10000,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            max_size: Maximum number of transitions to store
            device: Device for tensor operations
        """
        self.max_size = max_size
        self.device = device
        self.buffer: deque = deque(maxlen=max_size)

        # For GRPO: group transitions
        self.groups: List[List[Transition]] = []
        self._current_group: List[Transition] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition.detach())

    def add_to_group(self, transition: Transition) -> None:
        """Add a transition to the current group (for GRPO)."""
        self._current_group.append(transition.detach())

    def finish_group(self) -> None:
        """Finish the current group and start a new one."""
        if self._current_group:
            self.groups.append(self._current_group)
            self._current_group = []

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.buffer.clear()
        self.groups = []
        self._current_group = []

    def sample_batch(self, batch_size: int) -> List[Transition]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of Transition objects
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def sample_groups(self, num_groups: int) -> List[List[Transition]]:
        """
        Sample groups of transitions for GRPO.

        Args:
            num_groups: Number of groups to sample

        Returns:
            List of groups, each containing multiple transitions
        """
        if len(self.groups) < num_groups:
            return self.groups
        return random.sample(self.groups, num_groups)

    def get_all_groups(self) -> List[List[Transition]]:
        """Get all stored groups."""
        return self.groups

    def create_groups_from_buffer(self, group_size: int) -> List[List[Transition]]:
        """
        Create groups from the buffer by grouping consecutive transitions.

        Args:
            group_size: Number of transitions per group

        Returns:
            List of groups
        """
        buffer_list = list(self.buffer)
        groups = []
        for i in range(0, len(buffer_list), group_size):
            group = buffer_list[i:i + group_size]
            if len(group) == group_size:
                groups.append(group)
        return groups

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about stored transitions."""
        if len(self.buffer) == 0:
            return {
                "count": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_task_reward": 0.0,
                "mean_consistency_reward": 0.0,
                "mean_cost_penalty": 0.0,
            }

        rewards = [t.reward for t in self.buffer]
        task_rewards = [t.task_reward for t in self.buffer]
        consistency_rewards = [t.consistency_reward for t in self.buffer]
        cost_penalties = [t.cost_penalty for t in self.buffer]

        return {
            "count": len(self.buffer),
            "mean_reward": sum(rewards) / len(rewards),
            "std_reward": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5,
            "mean_task_reward": sum(task_rewards) / len(task_rewards),
            "mean_consistency_reward": sum(consistency_rewards) / len(consistency_rewards),
            "mean_cost_penalty": sum(cost_penalties) / len(cost_penalties),
        }


class BatchedTransition:
    """
    Batched transitions for efficient processing.

    Stacks multiple Transition objects into batched tensors.
    """

    def __init__(self, transitions: List[Transition]):
        """
        Args:
            transitions: List of Transition objects to batch
        """
        self.batch_size = len(transitions)

        # Stack tensors
        self.query_embeds = torch.stack([t.query_embed for t in transitions])
        self.selected_indices = torch.stack([t.selected_indices for t in transitions])
        self.log_probs = torch.stack([t.log_probs for t in transitions])

        # Handle variable-length block summaries
        max_blocks = max(t.block_summaries.size(0) for t in transitions)
        D = transitions[0].block_summaries.size(-1)
        self.block_summaries = torch.zeros(
            self.batch_size, max_blocks, D,
            device=transitions[0].block_summaries.device
        )
        self.block_mask = torch.zeros(
            self.batch_size, max_blocks,
            dtype=torch.bool,
            device=transitions[0].block_summaries.device
        )

        for i, t in enumerate(transitions):
            n_blocks = t.block_summaries.size(0)
            self.block_summaries[i, :n_blocks] = t.block_summaries
            self.block_mask[i, :n_blocks] = True

        # Stack rewards
        self.rewards = torch.tensor([t.reward for t in transitions])
        self.task_rewards = torch.tensor([t.task_reward for t in transitions])
        self.consistency_rewards = torch.tensor([t.consistency_reward for t in transitions])
        self.cost_penalties = torch.tensor([t.cost_penalty for t in transitions])

        # Store original probs if available
        if transitions[0].probs is not None:
            self.probs = torch.stack([
                torch.nn.functional.pad(t.probs, (0, max_blocks - t.probs.size(0)))
                for t in transitions
            ])
        else:
            self.probs = None

    def to(self, device: torch.device) -> 'BatchedTransition':
        """Move all tensors to specified device."""
        self.query_embeds = self.query_embeds.to(device)
        self.block_summaries = self.block_summaries.to(device)
        self.block_mask = self.block_mask.to(device)
        self.selected_indices = self.selected_indices.to(device)
        self.log_probs = self.log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.task_rewards = self.task_rewards.to(device)
        self.consistency_rewards = self.consistency_rewards.to(device)
        self.cost_penalties = self.cost_penalties.to(device)
        if self.probs is not None:
            self.probs = self.probs.to(device)
        return self
