"""
Heuristic-based memory block selection methods.

These serve as baselines to compare with the learned reading policy.
"""

import random
from typing import List, Tuple
import torch
import torch.nn.functional as F


class HeuristicSelector:
    """Base class for heuristic selection methods."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    def select_blocks(self, query_embed, block_summaries, **kwargs) -> List[int]:
        """
        Select top_k block indices.

        Args:
            query_embed: Query embedding [D] or [1, D]
            block_summaries: Block embeddings [num_blocks, D]
            **kwargs: Additional method-specific arguments

        Returns:
            List of selected block indices
        """
        raise NotImplementedError


class RandomSelector(HeuristicSelector):
    """Randomly select top_k blocks."""

    def __init__(self, top_k: int, seed: int = 42):
        super().__init__(top_k)
        self.rng = random.Random(seed)

    def select_blocks(self, query_embed, block_summaries, **kwargs) -> List[int]:
        num_blocks = block_summaries.shape[0]
        return self.rng.sample(range(num_blocks), min(self.top_k, num_blocks))


class RecencySelector(HeuristicSelector):
    """Select the most recent top_k blocks."""

    def select_blocks(self, query_embed, block_summaries, **kwargs) -> List[int]:
        num_blocks = block_summaries.shape[0]
        # Most recent blocks are at the end
        start_idx = max(0, num_blocks - self.top_k)
        return list(range(start_idx, num_blocks))


class SimilaritySelector(HeuristicSelector):
    """Select top_k blocks with highest cosine similarity to query."""

    def select_blocks(self, query_embed, block_summaries, **kwargs) -> List[int]:
        # Ensure correct shapes
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0)  # [1, D]
        if block_summaries.dim() == 2:
            block_summaries = block_summaries  # [num_blocks, D]

        # Compute cosine similarity
        # Normalize embeddings
        query_norm = F.normalize(query_embed, p=2, dim=-1)  # [1, D]
        blocks_norm = F.normalize(block_summaries, p=2, dim=-1)  # [num_blocks, D]

        # Similarity scores
        similarities = torch.matmul(blocks_norm, query_norm.T).squeeze(-1)  # [num_blocks]

        # Select top_k
        num_blocks = block_summaries.shape[0]
        k = min(self.top_k, num_blocks)
        _, indices = torch.topk(similarities, k)

        return indices.cpu().tolist()


class TimeWeightedSimilaritySelector(HeuristicSelector):
    """Combine recency and similarity with weights."""

    def __init__(self, top_k: int, similarity_weight: float = 0.7, recency_weight: float = 0.3):
        super().__init__(top_k)
        self.sim_weight = similarity_weight
        self.rec_weight = recency_weight
        assert abs(similarity_weight + recency_weight - 1.0) < 1e-6, "Weights must sum to 1"

    def select_blocks(self, query_embed, block_summaries, **kwargs) -> List[int]:
        # Ensure correct shapes
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0)  # [1, D]

        num_blocks = block_summaries.shape[0]

        # Similarity scores
        query_norm = F.normalize(query_embed, p=2, dim=-1)
        blocks_norm = F.normalize(block_summaries, p=2, dim=-1)
        sim_scores = torch.matmul(blocks_norm, query_norm.T).squeeze(-1)  # [num_blocks]

        # Recency scores (linear decay from 0 to 1)
        recency_scores = torch.linspace(0, 1, num_blocks).to(block_summaries.device)

        # Combined scores
        combined = self.sim_weight * sim_scores + self.rec_weight * recency_scores

        # Select top_k
        k = min(self.top_k, num_blocks)
        _, indices = torch.topk(combined, k)

        return indices.cpu().tolist()


def get_heuristic_selector(method: str, top_k: int, **kwargs):
    """
    Factory function to create heuristic selectors.

    Args:
        method: One of ['random', 'recency', 'similarity', 'time_weighted']
        top_k: Number of blocks to select
        **kwargs: Method-specific arguments

    Returns:
        HeuristicSelector instance
    """
    if method == 'random':
        return RandomSelector(top_k, **kwargs)
    elif method == 'recency':
        return RecencySelector(top_k)
    elif method == 'similarity':
        return SimilaritySelector(top_k)
    elif method == 'time_weighted':
        return TimeWeightedSimilaritySelector(top_k, **kwargs)
    else:
        raise ValueError(f"Unknown heuristic method: {method}")
