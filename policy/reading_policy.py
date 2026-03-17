"""
Reading Policy Network for LatentMAS.

Cross-attention based policy network that selects which memory blocks to read.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadingPolicyNetwork(nn.Module):
    """
    Cross-Attention based reading policy network.

    The policy takes a query embedding (from current agent/judger) and
    block summaries (from memory pool), and outputs selection probabilities
    for each block.

    Architecture:
    1. Project query embedding
    2. Multi-layer cross-attention: query attends to block summaries
    3. Score head: output selection probability for each block
    """

    def __init__(
        self,
        hidden_dim: int = 3584,  # Qwen3-14B hidden size
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden representations (should match model hidden size)
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Query projection (from current agent's embedding)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Multi-layer cross-attention
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Layer norms (one for each attention layer)
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None

        # Feed-forward layers after attention
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        if use_layer_norm:
            self.ffn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
        else:
            self.ffn_norms = None

        # Score head: output selection probability for each block
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        query_embed: torch.Tensor,
        block_summaries: torch.Tensor,
        temperature: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute selection probabilities for each block.

        Args:
            query_embed: Query embedding from current agent
                        Shape: [B, D] or [B, query_len, D]
            block_summaries: Summary vectors of all memory blocks
                            Shape: [B, num_blocks, D]
            temperature: Temperature for softmax (lower = sharper distribution)
            attention_mask: Optional mask for invalid blocks
                           Shape: [B, num_blocks], True = valid, False = masked

        Returns:
            logits: [B, num_blocks] raw scores before softmax
            probs: [B, num_blocks] selection probabilities
        """
        # Handle 2D query input: [B, D] -> [B, 1, D]
        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(1)

        B, num_blocks, D = block_summaries.shape

        # Project query
        query = self.query_proj(query_embed)  # [B, query_len, D]

        # Cross-attention layers
        # Query attends to block summaries (key/value)
        attn_output = query
        for i in range(self.num_layers):
            # Cross-attention: query -> block_summaries
            residual = attn_output

            if attention_mask is not None:
                # Convert to attention mask format for MultiheadAttention
                # True = valid -> False, False = masked -> True (inverted)
                key_padding_mask = ~attention_mask
            else:
                key_padding_mask = None

            attn_out, _ = self.cross_attn_layers[i](
                query=attn_output,
                key=block_summaries,
                value=block_summaries,
                key_padding_mask=key_padding_mask,
            )

            # Residual + LayerNorm
            attn_output = residual + attn_out
            if self.layer_norms is not None:
                attn_output = self.layer_norms[i](attn_output)

            # FFN
            residual = attn_output
            attn_output = residual + self.ffn_layers[i](attn_output)
            if self.ffn_norms is not None:
                attn_output = self.ffn_norms[i](attn_output)

        # Now we have context-aware query representation
        # Compute scores for each block by comparing with block summaries

        # Option 1: Use updated query to score each block
        # Expand query to match blocks: [B, 1, D] -> [B, num_blocks, D]
        query_expanded = attn_output.expand(-1, num_blocks, -1)

        # Combine query and block info for scoring
        # Simple approach: element-wise product followed by score head
        combined = query_expanded * block_summaries  # [B, num_blocks, D]

        # Score each block
        logits = self.score_head(combined).squeeze(-1)  # [B, num_blocks]

        # Apply temperature and mask
        logits = logits / temperature

        if attention_mask is not None:
            # Mask invalid blocks with large negative value
            logits = logits.masked_fill(~attention_mask, float('-inf'))

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def get_action_distribution(
        self,
        query_embed: torch.Tensor,
        block_summaries: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.distributions.Categorical:
        """
        Get categorical distribution over blocks for sampling.

        Args:
            query_embed: [B, D] or [B, query_len, D]
            block_summaries: [B, num_blocks, D]
            temperature: Temperature for softmax

        Returns:
            Categorical distribution over blocks
        """
        _, probs = self.forward(query_embed, block_summaries, temperature)
        return torch.distributions.Categorical(probs)


class LightweightReadingPolicy(nn.Module):
    """
    Simpler MLP-based reading policy for faster inference.

    This is a lighter alternative to the cross-attention policy,
    useful when computational budget is limited.
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        mlp_hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        # Block encoder
        self.block_encoder = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        # Score MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(mlp_hidden_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        query_embed: torch.Tensor,
        block_summaries: torch.Tensor,
        temperature: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embed: [B, D] or [B, query_len, D]
            block_summaries: [B, num_blocks, D]
            temperature: Temperature for softmax
            attention_mask: [B, num_blocks] mask

        Returns:
            logits: [B, num_blocks]
            probs: [B, num_blocks]
        """
        # Handle query dimensions
        if query_embed.dim() == 3:
            query_embed = query_embed.mean(dim=1)  # [B, D]

        B, num_blocks, D = block_summaries.shape

        # Encode query and blocks
        query_enc = self.query_encoder(query_embed)  # [B, mlp_hidden]
        block_enc = self.block_encoder(block_summaries)  # [B, num_blocks, mlp_hidden]

        # Expand query for concatenation
        query_exp = query_enc.unsqueeze(1).expand(-1, num_blocks, -1)  # [B, num_blocks, mlp_hidden]

        # Concatenate and score
        combined = torch.cat([query_exp, block_enc], dim=-1)  # [B, num_blocks, mlp_hidden*2]
        logits = self.score_mlp(combined).squeeze(-1)  # [B, num_blocks]

        # Apply temperature and mask
        logits = logits / temperature

        if attention_mask is not None:
            logits = logits.masked_fill(~attention_mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)

        return logits, probs
