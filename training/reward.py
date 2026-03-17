"""
Reward Calculator for RL-based reading policy in LatentMAS.

Computes multi-component rewards:
- R1: Task correctness (binary)
- R2: Evidence consistency (similarity-based)
- R3: Read cost penalty
"""

from typing import Dict, Optional, Tuple
import re

import torch
import torch.nn.functional as F


class RewardCalculator:
    """
    Computes rewards for the reading policy.

    Total reward: R = α*R1 + β*R2 - γ*R3

    Where:
    - R1: Task correctness (0 or 1)
    - R2: Evidence consistency (how well the answer depends on selected blocks)
    - R3: Read cost (penalty for reading more blocks/tokens)
    """

    def __init__(
        self,
        alpha: float = 1.0,    # Task correctness weight
        beta: float = 0.5,     # Evidence consistency weight
        gamma: float = 0.1,    # Read cost penalty weight
    ):
        """
        Args:
            alpha: Weight for task correctness reward
            beta: Weight for evidence consistency reward
            gamma: Weight for read cost penalty
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_task_reward(
        self,
        prediction: str,
        gold: str,
        task_type: str,
    ) -> float:
        """
        Compute R1: Task correctness reward.

        Args:
            prediction: Model's prediction
            gold: Ground truth answer
            task_type: Type of task (gsm8k, mbppplus, etc.)

        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        if prediction is None or gold is None:
            return 0.0

        prediction = str(prediction).strip().lower()
        gold = str(gold).strip().lower()

        # Numeric comparison for math tasks
        if task_type in ["gsm8k", "aime2024", "aime2025"]:
            try:
                pred_num = self._extract_number(prediction)
                gold_num = self._extract_number(gold)
                if pred_num is not None and gold_num is not None:
                    return 1.0 if abs(pred_num - gold_num) < 1e-6 else 0.0
            except (ValueError, TypeError):
                pass
            return 1.0 if prediction == gold else 0.0

        # Multiple choice tasks
        elif task_type in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            # Extract letter answer
            pred_letter = self._extract_letter(prediction)
            gold_letter = self._extract_letter(gold)
            return 1.0 if pred_letter == gold_letter else 0.0

        # Code tasks - handled externally with execution
        elif task_type in ["mbppplus", "humanevalplus"]:
            # For code tasks, correctness is determined by execution
            # This should be passed in directly
            return 1.0 if prediction == gold else 0.0

        # Default: exact match
        return 1.0 if prediction == gold else 0.0

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric answer from text."""
        # Try to find boxed answer first
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
        if boxed_match:
            content = boxed_match.group(1)
            number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
            if number:
                return float(number.group(0))

        # Fall back to last number in text
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if numbers:
            return float(numbers[-1])
        return None

    def _extract_letter(self, text: str) -> Optional[str]:
        """Extract letter answer (A/B/C/D) from text."""
        # Look in boxed format first
        boxed_match = re.search(r"\\boxed\{([ABCD])\}", text, re.IGNORECASE)
        if boxed_match:
            return boxed_match.group(1).upper()

        # Look for standalone letter at end
        letter_match = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

        return text.strip().upper() if len(text.strip()) == 1 else None

    def compute_evidence_consistency(
        self,
        final_hidden_states: torch.Tensor,
        selected_block_summaries: torch.Tensor,
        unselected_block_summaries: torch.Tensor,
    ) -> float:
        """
        Compute R2: Evidence consistency reward.

        Uses similarity difference method:
        - Compute similarity between final output and selected blocks
        - Compute similarity between final output and unselected blocks
        - R2 = sim_selected - sim_unselected

        Higher R2 means the output is more dependent on selected blocks.

        Args:
            final_hidden_states: [B, L, D] or [B, D] final hidden states from generation
            selected_block_summaries: [B, k, D] summaries of selected blocks
            unselected_block_summaries: [B, num_unsel, D] summaries of unselected blocks

        Returns:
            Evidence consistency score in [0, 1]
        """
        # Handle dimensions
        if final_hidden_states.dim() == 3:
            final_hidden_states = final_hidden_states.mean(dim=1)  # [B, D]
        final_hidden_states = final_hidden_states.unsqueeze(1)  # [B, 1, D]

        # Normalize for cosine similarity
        final_norm = F.normalize(final_hidden_states, dim=-1)
        selected_norm = F.normalize(selected_block_summaries, dim=-1)

        # Similarity with selected blocks: [B, 1, D] x [B, D, k] -> [B, 1, k]
        selected_sim = torch.bmm(
            final_norm,
            selected_norm.transpose(1, 2)
        ).squeeze(1)  # [B, k]
        avg_selected_sim = selected_sim.mean(dim=-1)  # [B]

        # Similarity with unselected blocks
        if unselected_block_summaries.size(1) > 0:
            unselected_norm = F.normalize(unselected_block_summaries, dim=-1)
            unselected_sim = torch.bmm(
                final_norm,
                unselected_norm.transpose(1, 2)
            ).squeeze(1)  # [B, num_unsel]
            avg_unselected_sim = unselected_sim.mean(dim=-1)  # [B]
        else:
            avg_unselected_sim = torch.zeros_like(avg_selected_sim)

        # Consistency score: higher when more focused on selected blocks
        # Transform from [-1, 1] to [0, 1]
        consistency_score = (avg_selected_sim - avg_unselected_sim + 1) / 2

        return float(consistency_score.mean().item())

    def compute_read_cost(
        self,
        num_selected_blocks: int,
        total_blocks: int,
        selected_kv_tokens: int = 0,
        total_kv_tokens: int = 0,
    ) -> float:
        """
        Compute R3: Read cost penalty.

        Penalizes reading more blocks/tokens than necessary.

        Args:
            num_selected_blocks: Number of blocks selected
            total_blocks: Total number of available blocks
            selected_kv_tokens: Number of KV tokens in selected blocks
            total_kv_tokens: Total number of KV tokens

        Returns:
            Read cost in [0, 1]
        """
        if total_blocks == 0:
            return 0.0

        block_ratio = num_selected_blocks / total_blocks

        if total_kv_tokens > 0 and selected_kv_tokens > 0:
            token_ratio = selected_kv_tokens / total_kv_tokens
            # Average of block and token ratios
            return (block_ratio + token_ratio) / 2
        else:
            return block_ratio

    def compute_total_reward(
        self,
        prediction: str,
        gold: str,
        task_type: str,
        final_hidden_states: Optional[torch.Tensor] = None,
        selected_block_summaries: Optional[torch.Tensor] = None,
        unselected_block_summaries: Optional[torch.Tensor] = None,
        num_selected_blocks: int = 0,
        total_blocks: int = 0,
        selected_kv_tokens: int = 0,
        total_kv_tokens: int = 0,
        correct_override: Optional[bool] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward: R = α*R1 + β*R2 - γ*R3

        Args:
            prediction: Model's prediction
            gold: Ground truth answer
            task_type: Type of task
            final_hidden_states: Final hidden states from generation
            selected_block_summaries: Summaries of selected blocks
            unselected_block_summaries: Summaries of unselected blocks
            num_selected_blocks: Number of blocks selected
            total_blocks: Total number of available blocks
            selected_kv_tokens: Number of KV tokens in selected blocks
            total_kv_tokens: Total number of KV tokens
            correct_override: Optional override for task correctness (for code execution)

        Returns:
            (total_reward, component_dict) where component_dict contains individual rewards
        """
        # R1: Task correctness
        if correct_override is not None:
            r1 = 1.0 if correct_override else 0.0
        else:
            r1 = self.compute_task_reward(prediction, gold, task_type)

        # R2: Evidence consistency
        if (final_hidden_states is not None and
            selected_block_summaries is not None and
            unselected_block_summaries is not None):
            r2 = self.compute_evidence_consistency(
                final_hidden_states,
                selected_block_summaries,
                unselected_block_summaries,
            )
        else:
            r2 = 0.5  # Neutral if not available

        # R3: Read cost
        r3 = self.compute_read_cost(
            num_selected_blocks,
            total_blocks,
            selected_kv_tokens,
            total_kv_tokens,
        )

        # Total reward
        total = self.alpha * r1 + self.beta * r2 - self.gamma * r3

        components = {
            "task_reward": r1,
            "consistency_reward": r2,
            "cost_penalty": r3,
            "total_reward": total,
        }

        return total, components

    def compute_batch_rewards(
        self,
        predictions: list,
        golds: list,
        task_type: str,
        correct_flags: Optional[list] = None,
        num_selected_list: Optional[list] = None,
        total_blocks_list: Optional[list] = None,
    ) -> Tuple[list, list]:
        """
        Compute rewards for a batch of predictions.

        Simplified version without hidden states (for efficiency).

        Args:
            predictions: List of model predictions
            golds: List of ground truth answers
            task_type: Type of task
            correct_flags: Optional list of correctness flags (for code execution)
            num_selected_list: List of number of selected blocks
            total_blocks_list: List of total blocks

        Returns:
            (rewards, components_list)
        """
        rewards = []
        components_list = []

        for i in range(len(predictions)):
            correct_override = correct_flags[i] if correct_flags else None
            num_sel = num_selected_list[i] if num_selected_list else 0
            total = total_blocks_list[i] if total_blocks_list else 0

            reward, components = self.compute_total_reward(
                prediction=predictions[i],
                gold=golds[i],
                task_type=task_type,
                num_selected_blocks=num_sel,
                total_blocks=total,
                correct_override=correct_override,
            )
            rewards.append(reward)
            components_list.append(components)

        return rewards, components_list


class AdaptiveRewardCalculator(RewardCalculator):
    """
    Adaptive reward calculator that adjusts weights based on training progress.

    Initially focuses on task correctness, then gradually increases
    emphasis on efficiency (read cost).
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        alpha_final: float = 1.0,
        beta_init: float = 0.3,
        beta_final: float = 0.5,
        gamma_init: float = 0.0,
        gamma_final: float = 0.2,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
    ):
        """
        Args:
            alpha_init/final: Task correctness weight (start/end)
            beta_init/final: Evidence consistency weight (start/end)
            gamma_init/final: Read cost penalty weight (start/end)
            warmup_steps: Steps before starting to adjust weights
            total_steps: Total training steps for full weight adjustment
        """
        super().__init__(alpha_init, beta_init, gamma_init)

        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> None:
        """Update weights based on training progress."""
        self.current_step += 1
        progress = self._get_progress()

        self.alpha = self._interpolate(self.alpha_init, self.alpha_final, progress)
        self.beta = self._interpolate(self.beta_init, self.beta_final, progress)
        self.gamma = self._interpolate(self.gamma_init, self.gamma_final, progress)

    def _get_progress(self) -> float:
        """Get training progress as fraction in [0, 1]."""
        if self.current_step < self.warmup_steps:
            return 0.0
        effective_step = self.current_step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps
        return min(1.0, effective_step / max(1, effective_total))

    def _interpolate(self, start: float, end: float, progress: float) -> float:
        """Linear interpolation between start and end."""
        return start + (end - start) * progress

    def get_current_weights(self) -> Dict[str, float]:
        """Get current reward weights."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "progress": self._get_progress(),
        }
