"""
Block Selector for RL-based reading policy in LatentMAS.

Provides utilities for selecting top-k blocks with different strategies
suitable for training (sampling) and inference (deterministic).
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class BlockSelector:
    """
    Utility class for selecting memory blocks based on policy probabilities.

    Supports multiple selection strategies:
    - Categorical sampling (for RL training)
    - Top-k deterministic (for inference)
    - Gumbel-softmax (differentiable sampling)
    - Nucleus (top-p) sampling
    """

    def __init__(
        self,
        default_k: int = 4,
        temperature: float = 1.0,
        min_prob_threshold: float = 1e-8,
    ):
        """
        Args:
            default_k: Default number of blocks to select
            temperature: Temperature for sampling
            min_prob_threshold: Minimum probability threshold to avoid numerical issues
        """
        self.default_k = default_k
        self.temperature = temperature
        self.min_prob_threshold = min_prob_threshold

    def select_top_k(
        self,
        probs: torch.Tensor,
        k: Optional[int] = None,
        training: bool = True,
        return_log_probs: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Select top-k blocks based on probabilities.

        Args:
            probs: [B, num_blocks] selection probabilities
            k: Number of blocks to select (defaults to self.default_k)
            training: If True, use sampling; if False, use deterministic top-k
            return_log_probs: Whether to return log probabilities

        Returns:
            selected_indices: [B, k] selected block indices
            log_probs: [B, k] log probabilities (if return_log_probs=True)
        """
        if k is None:
            k = self.default_k

        B, num_blocks = probs.shape
        k = min(k, num_blocks)  # Can't select more than available

        if k == 0:
            empty = torch.zeros(B, 0, dtype=torch.long, device=probs.device)
            return (empty, torch.zeros(B, 0, device=probs.device)) if return_log_probs else (empty, None)

        if training:
            return self._sample_k(probs, k, return_log_probs)
        else:
            return self._top_k_deterministic(probs, k, return_log_probs)

    def _sample_k(
        self,
        probs: torch.Tensor,
        k: int,
        return_log_probs: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample k blocks without replacement using categorical sampling.

        Uses iterative sampling with probability masking to ensure
        no duplicates.
        """
        B, num_blocks = probs.shape
        device = probs.device

        # Convert to float32 for numerical stability in sampling
        original_dtype = probs.dtype
        probs = probs.float()

        # Clamp probabilities for numerical stability
        probs = probs.clamp(min=self.min_prob_threshold)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        selected_indices = []
        selected_log_probs = []

        # Working copy of probabilities
        current_probs = probs.clone()

        for _ in range(k):
            # Renormalize to ensure valid probability distribution
            current_probs = current_probs.clamp(min=0)
            prob_sum = current_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            current_probs = current_probs / prob_sum

            # Sample from current distribution (disable validation for numerical stability)
            dist = torch.distributions.Categorical(current_probs, validate_args=False)
            idx = dist.sample()  # [B]

            if return_log_probs:
                log_prob = dist.log_prob(idx)  # [B]
                selected_log_probs.append(log_prob)

            selected_indices.append(idx)

            # Zero out selected indices for sampling without replacement
            mask = torch.zeros_like(current_probs)
            mask.scatter_(1, idx.unsqueeze(1), 1.0)
            current_probs = current_probs * (1 - mask)

        selected = torch.stack(selected_indices, dim=1)  # [B, k]

        if return_log_probs:
            log_probs = torch.stack(selected_log_probs, dim=1)  # [B, k]
            return selected, log_probs
        return selected, None

    def _top_k_deterministic(
        self,
        probs: torch.Tensor,
        k: int,
        return_log_probs: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Select top-k blocks deterministically.
        """
        _, indices = probs.topk(k, dim=-1)  # [B, k]

        if return_log_probs:
            log_probs = probs.gather(dim=-1, index=indices).log()
            return indices, log_probs
        return indices, None

    def sample_gumbel_softmax(
        self,
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gumbel-Softmax sampling for differentiable selection.

        This allows gradients to flow through the selection process.

        Args:
            logits: [B, num_blocks] unnormalized scores
            k: Number of blocks to select
            temperature: Gumbel-Softmax temperature
            hard: If True, use straight-through estimator

        Returns:
            selected_indices: [B, k] selected block indices
            soft_selection: [B, k, num_blocks] soft selection weights
        """
        B, num_blocks = logits.shape
        k = min(k, num_blocks)

        selected_indices = []
        soft_selections = []

        current_logits = logits.clone()

        for _ in range(k):
            # Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(current_logits) + 1e-10
            ) + 1e-10)

            soft = F.softmax((current_logits + gumbel_noise) / temperature, dim=-1)

            if hard:
                # Straight-through estimator
                idx = soft.argmax(dim=-1)
                hard_one_hot = F.one_hot(idx, num_blocks).float()
                soft = hard_one_hot - soft.detach() + soft
            else:
                idx = soft.argmax(dim=-1)

            selected_indices.append(idx)
            soft_selections.append(soft)

            # Mask selected index
            mask = F.one_hot(idx, num_blocks).float()
            current_logits = current_logits - 1e10 * mask

        selected = torch.stack(selected_indices, dim=1)  # [B, k]
        soft_selection = torch.stack(soft_selections, dim=1)  # [B, k, num_blocks]

        return selected, soft_selection

    def sample_nucleus(
        self,
        probs: torch.Tensor,
        top_p: float = 0.9,
        max_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nucleus (top-p) sampling - select blocks until cumulative prob exceeds top_p.

        Args:
            probs: [B, num_blocks] selection probabilities
            top_p: Cumulative probability threshold
            max_k: Maximum number of blocks to select

        Returns:
            selected_indices: [B, actual_k] selected block indices (variable length)
            log_probs: [B, actual_k] log probabilities
        """
        B, num_blocks = probs.shape

        # Sort by probability (descending)
        sorted_probs, sorted_indices = probs.sort(descending=True, dim=-1)

        # Compute cumulative probabilities
        cumsum = sorted_probs.cumsum(dim=-1)

        # Find cutoff index where cumsum exceeds top_p
        mask = cumsum <= top_p
        # Include at least one and the first one that exceeds
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = True

        if max_k is not None:
            max_mask = torch.zeros_like(mask)
            max_mask[:, :max_k] = True
            mask = mask & max_mask

        # For simplicity, use the maximum selected count across batch
        # (In practice, you might want variable-length outputs)
        k = mask.sum(dim=-1).max().item()
        k = max(1, k)

        selected_indices = sorted_indices[:, :k]
        selected_probs = sorted_probs[:, :k]

        return selected_indices, selected_probs.log()

    @staticmethod
    def compute_selection_entropy(probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the selection distribution.

        Higher entropy = more uniform distribution = more exploration.

        Args:
            probs: [B, num_blocks] selection probabilities

        Returns:
            [B] entropy values
        """
        # Avoid log(0)
        probs_clamped = probs.clamp(min=1e-10)
        entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        return entropy

    @staticmethod
    def compute_kl_divergence(
        probs: torch.Tensor,
        ref_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference distributions.

        Args:
            probs: [B, num_blocks] current probabilities
            ref_probs: [B, num_blocks] reference probabilities

        Returns:
            [B] KL divergence values
        """
        probs_clamped = probs.clamp(min=1e-10)
        ref_probs_clamped = ref_probs.clamp(min=1e-10)
        kl = (probs_clamped * (probs_clamped.log() - ref_probs_clamped.log())).sum(dim=-1)
        return kl
