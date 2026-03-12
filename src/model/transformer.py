"""
GPT (Decoder-Only Transformer) — built from scratch.

Following Chapter 4 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

Architecture:
  Token Embeddings + Positional Embeddings
  → Dropout
  → N × TransformerBlock
        ├─ LayerNorm → MultiHeadAttention → Residual
        └─ LayerNorm → FeedForward        → Residual
  → LayerNorm
  → Linear output head  (vocab_size logits)
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention


# ──────────────────────────────────────────────
# Default configuration (GPT-small for Arabic math)
# ──────────────────────────────────────────────

GPT_CONFIG = {
    "vocab_size": 5000,
    "context_length": 256,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# ──────────────────────────────────────────────
# Layer Normalization (from scratch, no built-in)
# ──────────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    Layer Normalization (Ba et al., 2016).

    Normalizes across the last dimension (feature dim) for each token
    independently.  Uses learnable scale (γ) and shift (β).
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))   # γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))   # β

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift


# ──────────────────────────────────────────────
# GELU activation
# ──────────────────────────────────────────────

class GELU(nn.Module):
    """Gaussian Error Linear Unit (used in GPT-2 / GPT-3)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            (2.0 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)
        ))


# ──────────────────────────────────────────────
# Feed-Forward Network
# ──────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear layers with GELU activation in between.
    Expands d_model → 4·d_model → d_model.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ──────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block (pre-norm variant).

    Architecture:
        x  ─→  LayerNorm → MHA   → + (residual)
           ─→  LayerNorm → FFN   → + (residual)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.attn = MultiHeadAttention(
            d_model=cfg["emb_dim"],
            n_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
        )
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.ffn = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        # Self-attention with residual
        normed = self.ln1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(normed, return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(normed)

        # Feed-forward with residual
        x = x + self.ffn(self.ln2(x))

        if return_attention:
            return x, attn_weights
        return x


# ──────────────────────────────────────────────
# GPT Model
# ──────────────────────────────────────────────

class GPTModel(nn.Module):
    """
    GPT — Generative Pre-trained Transformer (decoder-only).

    Components:
        1. Token embedding   (vocab_size → emb_dim)
        2. Positional embedding (context_length → emb_dim)
        3. Dropout
        4. N × TransformerBlock
        5. Final LayerNorm
        6. Linear output head (emb_dim → vocab_size)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying (same matrix for input embeddings & output head)
        self.out_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization following GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            idx: (B, T) token indices
            return_attention: if True, also return per-layer attention weights

        Returns:
            logits: (B, T, vocab_size)
            attn_maps: list of (B, n_heads, T, T) — one per layer (optional)
        """
        B, T = idx.shape
        assert T <= self.cfg["context_length"], \
            f"Sequence length {T} exceeds context_length {self.cfg['context_length']}"

        tok_emb = self.tok_emb(idx)                                     # (B, T, C)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))      # (T, C)
        x = self.drop_emb(tok_emb + pos_emb)

        attn_maps = []
        for block in self.trf_blocks:
            if return_attention:
                x, attn_w = block(x, return_attention=True)
                attn_maps.append(attn_w)
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)                                       # (B, T, V)

        if return_attention:
            return logits, attn_maps
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> list[dict]:
        """Return structured info about each layer (for demo visualization)."""
        info = []
        info.append({
            "name": "Token Embedding",
            "shape": f"{self.cfg['vocab_size']} × {self.cfg['emb_dim']}",
            "params": self.tok_emb.weight.numel(),
        })
        info.append({
            "name": "Positional Embedding",
            "shape": f"{self.cfg['context_length']} × {self.cfg['emb_dim']}",
            "params": self.pos_emb.weight.numel(),
        })
        for i, block in enumerate(self.trf_blocks):
            block_params = sum(p.numel() for p in block.parameters())
            info.append({
                "name": f"Transformer Block {i}",
                "shape": f"{self.cfg['emb_dim']}d, {self.cfg['n_heads']} heads",
                "params": block_params,
            })
        info.append({
            "name": "Output Head (tied)",
            "shape": f"{self.cfg['emb_dim']} → {self.cfg['vocab_size']}",
            "params": 0,
        })
        return info


# ──────────────────────────────────────────────
# Text Generation Utilities
# ──────────────────────────────────────────────

def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressive generation with temperature scaling and top-k sampling.

    Args:
        model:          trained GPTModel
        idx:            (B, T) initial token indices
        max_new_tokens: how many tokens to generate
        context_length: model's maximum context window
        temperature:    softmax temperature (< 1 = sharper, > 1 = flatter)
        top_k:          keep only top-k logits before sampling
        eos_id:         stop generation when this token is produced

    Returns:
        idx: (B, T + max_new_tokens) — original + generated tokens
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx[:, -context_length:]

            logits = model(idx_cond)              # (B, T, V)
            logits = logits[:, -1, :]              # last position  (B, V)

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                min_val = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_val,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=-1)

            if eos_id is not None and (next_token == eos_id).all():
                break

    return idx


def generate_step_by_step(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = 25,
) -> list[dict]:
    """
    Generate tokens one-by-one, returning detailed info at each step.
    Useful for the demo to visualize the generation process.
    """
    model.eval()
    steps = []
    with torch.no_grad():
        for step in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
            logits, attn_maps = model(idx_cond, return_attention=True)
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                min_val = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_val,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Top-5 candidates
            top5_probs, top5_ids = torch.topk(probs, 5)

            steps.append({
                "step": step,
                "token_id": next_token.item(),
                "probability": probs[0, next_token.item()].item(),
                "top5_ids": top5_ids[0].tolist(),
                "top5_probs": top5_probs[0].tolist(),
                "last_layer_attn": attn_maps[-1][0, 0].cpu(),  # head 0 of last layer
                "sequence_so_far": idx[0].tolist() + [next_token.item()],
            })

            idx = torch.cat([idx, next_token], dim=-1)

    return steps
