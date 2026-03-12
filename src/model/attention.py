"""
Attention Mechanisms — built from scratch.

Following Chapter 3 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

Implements:
  1. Scaled Dot-Product Attention  (the fundamental operation)
  2. CausalSelfAttention           (single-head with causal mask)
  3. MultiHeadAttention            (parallel heads + output projection)
"""

import torch
import torch.nn as nn
import math


# ──────────────────────────────────────────────
# 1. Scaled Dot-Product Attention (functional)
# ──────────────────────────────────────────────

def scaled_dot_product_attention(
    query: torch.Tensor,   # (B, n_heads, T, d_k)
    key: torch.Tensor,     # (B, n_heads, T, d_k)
    value: torch.Tensor,   # (B, n_heads, T, d_k)
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Returns:
        context  : (B, n_heads, T, d_k)
        weights  : (B, n_heads, T, T)   — attention weights (for visualization)
    """
    d_k = query.size(-1)
    # QK^T / sqrt(d_k)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply causal mask: future positions → -inf
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    context = torch.matmul(attn_weights, value)
    return context, attn_weights


# ──────────────────────────────────────────────
# 2. Causal Self-Attention (single head)
# ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Single-head causal (masked) self-attention.

    Used as a building block to explain the concept before scaling
    to multi-head attention.
    """

    def __init__(self, d_model: int, context_length: int, dropout: float = 0.0):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper-triangular = 0)
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))  # (1,1,T,T)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        q = self.W_q(x).unsqueeze(1)  # (B,1,T,C)
        k = self.W_k(x).unsqueeze(1)
        v = self.W_v(x).unsqueeze(1)

        context, weights = scaled_dot_product_attention(
            q, k, v, mask=self.mask[:, :, :T, :T], dropout=self.dropout
        )
        return context.squeeze(1), weights.squeeze(1)


# ──────────────────────────────────────────────
# 3. Multi-Head Attention
# ──────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with causal masking.

    Splits d_model into n_heads parallel attention heads, each of
    dimension d_k = d_model // n_heads, then concatenates and projects.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # QKV projections (combined for efficiency)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))  # (1,1,T,T)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        # Project to Q, K, V  →  reshape into (B, n_heads, T, d_k)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        context, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=self.mask[:, :, :T, :T], dropout=self.dropout
        )

        # Concatenate heads  →  (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.out_proj(context)

        if return_attention:
            return output, attn_weights
        return output
