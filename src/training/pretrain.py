"""
Pretraining Loop — self-supervised next-token prediction.

Following Chapter 5 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

Pipeline:
  1. Load tokenizer & build DataLoader (sliding-window chunks)
  2. Initialize GPTModel
  3. Train with cross-entropy loss + AdamW optimiser
  4. Log training/validation loss, save checkpoints
  5. Generate sample text periodically to monitor quality
"""

import os
import sys
import time
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.transformer import GPTModel, GPT_CONFIG, generate
from src.training.prepare_data import (
    create_pretrain_dataloader,
    PretrainDataset,
)


# ──────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────

def calc_loss_batch(model, input_batch, target_batch, device):
    """Cross-entropy loss for a single batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)                            # (B, T, V)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1),
    )
    return loss


def calc_loss_loader(model, data_loader, device, max_batches=None):
    """Average loss across a data loader."""
    total_loss, num_batches = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            loss = calc_loss_batch(model, inp, tgt, device)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)


# ──────────────────────────────────────────────
# Learning rate scheduler (cosine with warmup)
# ──────────────────────────────────────────────

def cosine_lr_schedule(optimizer, step, total_steps, lr_max, lr_min=1e-6, warmup_steps=100):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        lr = lr_max * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def pretrain(
    model: GPTModel,
    train_loader,
    val_loader,
    tokenizer: BPETokenizer,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    checkpoint_dir: str | None = None,
    log_interval: int = 50,
    eval_interval: int = 200,
    sample_interval: int = 500,
    sample_prompt: str = "حل المساله",
) -> dict:
    """
    Full pretraining loop.

    Returns a history dict with losses and generated samples.
    """
    checkpoint_dir = Path(checkpoint_dir or PROJECT_ROOT / "checkpoints" / "pretrained")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = num_epochs * len(train_loader)
    warmup_steps = min(100, total_steps // 10)

    history = {
        "train_losses": [],
        "val_losses": [],
        "lrs": [],
        "samples": [],
        "steps": [],
    }

    model.to(device)
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  Pretraining — {model.count_parameters():,} parameters")
    print(f"  Device: {device}  |  Epochs: {num_epochs}  |  Steps: {total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        for batch_idx, (inp, tgt) in enumerate(train_loader):
            # LR schedule
            current_lr = cosine_lr_schedule(
                optimizer, global_step, total_steps, lr, warmup_steps=warmup_steps
            )

            # Forward + backward
            loss = calc_loss_batch(model, inp, tgt, device)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            global_step += 1

            # ── Logging ───────────────────────
            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                history["train_losses"].append(loss.item())
                history["lrs"].append(current_lr)
                history["steps"].append(global_step)
                print(f"  Epoch {epoch+1}/{num_epochs} | "
                      f"Step {global_step}/{total_steps} | "
                      f"Loss {loss.item():.4f} | "
                      f"LR {current_lr:.2e} | "
                      f"Time {elapsed:.0f}s")

            # ── Validation ────────────────────
            if global_step % eval_interval == 0:
                val_loss = calc_loss_loader(model, val_loader, device, max_batches=20)
                history["val_losses"].append(val_loss)
                print(f"    → Val Loss: {val_loss:.4f}")
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": model.cfg,
                        "step": global_step,
                        "val_loss": val_loss,
                    }, checkpoint_dir / "best_model.pt")
                    print(f"    ✓ Best model saved (val_loss={val_loss:.4f})")

            # ── Sample generation ─────────────
            if global_step % sample_interval == 0:
                sample_ids = tokenizer.encode(sample_prompt)
                sample_tensor = torch.tensor([sample_ids], device=device)
                gen_ids = generate(
                    model, sample_tensor,
                    max_new_tokens=50,
                    context_length=model.cfg["context_length"],
                    temperature=0.8,
                    top_k=25,
                )
                gen_text = tokenizer.decode(gen_ids[0].tolist())
                history["samples"].append({
                    "step": global_step,
                    "text": gen_text,
                })
                print(f"    Sample: {gen_text[:120]}...")

    # Final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.cfg,
        "step": global_step,
        "val_loss": best_val_loss,
    }, checkpoint_dir / "final_model.pt")

    # Save history
    with open(checkpoint_dir / "pretrain_history.json", "w") as f:
        # Remove non-serializable tensors from samples
        json.dump(history, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\nPretraining complete in {total_time:.0f}s")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {checkpoint_dir}")

    return history


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def run_pretrain(
    pretrain_text_path: str | None = None,
    tokenizer_path: str | None = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-4,
    context_length: int = 256,
    device: str | None = None,
):
    """End-to-end pretrain: load data → train → save."""
    pretrain_text_path = pretrain_text_path or str(PROJECT_ROOT / "data/pretrain/data.txt")
    tokenizer_path = tokenizer_path or str(PROJECT_ROOT / "data/tokenizer.json")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device)

    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)

    # Build dataset & split
    dataset = PretrainDataset(pretrain_text_path, tokenizer, context_length)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Dataset: {len(dataset)} samples  (train={train_size}, val={val_size})")

    # Build model
    cfg = dict(GPT_CONFIG)
    cfg["vocab_size"] = tokenizer.vocab_size
    cfg["context_length"] = context_length
    model = GPTModel(cfg)
    print(f"Model: {model.count_parameters():,} parameters")

    history = pretrain(
        model, train_loader, val_loader, tokenizer,
        device=device, num_epochs=num_epochs, lr=lr,
    )
    return model, tokenizer, history


if __name__ == "__main__":
    run_pretrain()
