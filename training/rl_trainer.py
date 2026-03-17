"""
GRPO (Group Relative Policy Optimization) Trainer for LatentMAS.

Reference: DeepSeek-R1 技术报告
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .trajectory import Transition, TrajectoryBuffer, BatchedTransition
from .reward import RewardCalculator


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    GRPO advantages:
    - No value network required (unlike PPO)
    - Uses group-relative advantages: advantage_i = (r_i - mean(group)) / std(group)
    - More stable training for discrete action spaces

    Training loop:
    1. Collect trajectories in groups (same question, different selections)
    2. Compute group-relative advantages
    3. Update policy using clipped objective with KL penalty
    """

    def __init__(
        self,
        policy_net: nn.Module,
        optimizer: Optimizer,
        reward_calculator: RewardCalculator,
        group_size: int = 8,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        """
        Args:
            policy_net: The reading policy network to train
            optimizer: Optimizer for the policy network
            reward_calculator: Calculator for computing rewards
            group_size: Number of samples per group for relative advantage
            clip_epsilon: PPO-style clipping parameter
            entropy_coef: Coefficient for entropy bonus
            kl_coef: Coefficient for KL divergence penalty
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for tensor operations
            scheduler: Optional learning rate scheduler
        """
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.reward_calculator = reward_calculator
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or next(policy_net.parameters()).device
        self.scheduler = scheduler

        # Reference policy for KL computation
        self.ref_policy = copy.deepcopy(policy_net)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(device=self.device)

        # Training statistics
        self.global_step = 0
        self.stats_history: List[Dict] = []

    def compute_group_advantage(self, group_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-relative advantage.

        advantage_i = (r_i - mean(group_rewards)) / std(group_rewards)

        Args:
            group_rewards: [group_size] tensor of rewards

        Returns:
            [group_size] tensor of advantages
        """
        mean_r = group_rewards.mean()
        std_r = group_rewards.std().clamp(min=1e-6)
        advantages = (group_rewards - mean_r) / std_r
        return advantages

    def update_policy(
        self,
        groups: List[List[Transition]],
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """
        Update policy using GRPO on collected groups.

        Args:
            groups: List of groups, each containing transitions
            num_epochs: Number of passes over the data

        Returns:
            Dictionary of training statistics
        """
        self.policy_net.train()

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        num_updates = 0

        for epoch in range(num_epochs):
            for group in groups:
                if len(group) < 2:
                    continue  # Need at least 2 samples for relative advantage

                # Get rewards and compute advantages
                rewards = torch.tensor(
                    [t.reward for t in group],
                    device=self.device,
                    dtype=torch.float32
                )
                advantages = self.compute_group_advantage(rewards)

                group_policy_loss = 0.0
                group_kl_loss = 0.0
                group_entropy_loss = 0.0

                for i, transition in enumerate(group):
                    transition = transition.to(self.device)

                    # Forward pass through current policy
                    _, probs = self.policy_net(
                        transition.query_embed.unsqueeze(0),
                        transition.block_summaries.unsqueeze(0),
                    )

                    # Get log probs for selected indices
                    curr_log_probs = probs.log()
                    selected_log_prob = curr_log_probs.gather(
                        -1, transition.selected_indices.unsqueeze(0)
                    ).sum()

                    # Reference policy log probs for KL
                    with torch.no_grad():
                        _, ref_probs = self.ref_policy(
                            transition.query_embed.unsqueeze(0),
                            transition.block_summaries.unsqueeze(0),
                        )
                        ref_log_probs = ref_probs.log()
                        ref_selected_log_prob = ref_log_probs.gather(
                            -1, transition.selected_indices.unsqueeze(0)
                        ).sum()

                    # Original log prob from collection
                    old_log_prob = transition.log_probs.sum()

                    # Compute ratio
                    ratio = (selected_log_prob - old_log_prob).exp()

                    # Clipped objective
                    adv = advantages[i]
                    surr1 = ratio * adv
                    surr2 = ratio.clamp(
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon
                    ) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    # KL divergence penalty
                    kl_div = (ref_selected_log_prob - selected_log_prob).detach()
                    kl_loss = self.kl_coef * kl_div.abs()

                    # Entropy bonus
                    entropy = -(probs * probs.log().clamp(min=-100)).sum()
                    entropy_loss = -self.entropy_coef * entropy

                    group_policy_loss += policy_loss
                    group_kl_loss += kl_loss
                    group_entropy_loss += entropy_loss

                # Average over group
                n = len(group)
                group_policy_loss /= n
                group_kl_loss /= n
                group_entropy_loss /= n

                # Total loss for this group
                loss = group_policy_loss + group_kl_loss + group_entropy_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_net.parameters(),
                        self.max_grad_norm
                    )

                self.optimizer.step()

                # Accumulate statistics
                total_policy_loss += group_policy_loss.item()
                total_kl_loss += group_kl_loss.item()
                total_entropy_loss += group_entropy_loss.item()
                total_loss += loss.item()
                num_updates += 1

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Compute averages
        if num_updates > 0:
            stats = {
                "policy_loss": total_policy_loss / num_updates,
                "kl_loss": total_kl_loss / num_updates,
                "entropy_loss": total_entropy_loss / num_updates,
                "total_loss": total_loss / num_updates,
                "num_groups": len(groups),
                "global_step": self.global_step,
            }
        else:
            stats = {
                "policy_loss": 0.0,
                "kl_loss": 0.0,
                "entropy_loss": 0.0,
                "total_loss": 0.0,
                "num_groups": 0,
                "global_step": self.global_step,
            }

        self.stats_history.append(stats)
        return stats

    def update_reference_policy(self) -> None:
        """Update reference policy to current policy."""
        self.ref_policy.load_state_dict(self.policy_net.state_dict())

    def should_update_reference(self, update_freq: int = 100) -> bool:
        """Check if reference policy should be updated."""
        return self.global_step > 0 and self.global_step % update_freq == 0

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.add(transition)

    def add_to_current_group(self, transition: Transition) -> None:
        """Add a transition to the current group."""
        self.buffer.add_to_group(transition)

    def finish_current_group(self) -> None:
        """Finish the current group and prepare for next."""
        self.buffer.finish_group()

    def get_groups(self) -> List[List[Transition]]:
        """Get all collected groups."""
        return self.buffer.get_all_groups()

    def clear_buffer(self) -> None:
        """Clear the trajectory buffer."""
        self.buffer.clear()

    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        if not self.stats_history:
            return {}
        return self.stats_history[-1]

    def save_checkpoint(self, path: str) -> None:
        """Save trainer checkpoint."""
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "ref_policy_state_dict": self.ref_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "stats_history": self.stats_history,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load trainer checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.ref_policy.load_state_dict(checkpoint["ref_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.stats_history = checkpoint.get("stats_history", [])


class OnlineGRPOTrainer(GRPOTrainer):
    """
    Online GRPO trainer that updates policy after each group.

    Suitable for continuous training during evaluation.
    """

    def __init__(
        self,
        *args,
        update_every_n_groups: int = 4,
        **kwargs
    ):
        """
        Args:
            update_every_n_groups: Update policy after collecting this many groups
            *args, **kwargs: Passed to GRPOTrainer
        """
        super().__init__(*args, **kwargs)
        self.update_every_n_groups = update_every_n_groups
        self._pending_groups: List[List[Transition]] = []

    def add_group(self, group: List[Transition]) -> Optional[Dict[str, float]]:
        """
        Add a completed group and potentially trigger update.

        Args:
            group: List of transitions forming a group

        Returns:
            Training stats if update was performed, None otherwise
        """
        self._pending_groups.append(group)

        if len(self._pending_groups) >= self.update_every_n_groups:
            stats = self.update_policy(self._pending_groups)
            self._pending_groups = []
            return stats

        return None

    def flush(self) -> Optional[Dict[str, float]]:
        """Force update with remaining pending groups."""
        if self._pending_groups:
            stats = self.update_policy(self._pending_groups)
            self._pending_groups = []
            return stats
        return None
