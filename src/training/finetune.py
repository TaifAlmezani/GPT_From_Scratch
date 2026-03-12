"""
Fine-Tuning — Classification head + Instruction SFT with loss masking.

Following Chapters 6-7 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

Chapter 6: Transfer learning, classification head, freezing vs full fine-tuning
Chapter 7: Instruction datasets, prompt formatting, supervised fine-tuning (SFT), loss masking

Implements:
  1. GPTClassifier — adds a classification head on top of the pretrained GPT
  2. Instruction SFT — fine-tunes the full model on instruction-response pairs
     with loss masking so the model only learns to predict the response
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.transformer import GPTModel, GPT_CONFIG, generate
from src.training.prepare_data import SFTDataset


# ──────────────────────────────────────────────
# Chapter 6: Classification Head
# ──────────────────────────────────────────────

class GPTClassifier(nn.Module):
    """
    GPT with a classification head for transfer learning.

    Takes a pretrained GPT, freezes (optionally) the transformer blocks,
    and adds a linear classification head on top of the last token's
    hidden state.
    """

    def __init__(self, base_model: GPTModel, num_classes: int, freeze_base: bool = True):
        super().__init__()
        self.base_model = deepcopy(base_model)
        emb_dim = base_model.cfg["emb_dim"]

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze last transformer block + final norm
            for param in self.base_model.trf_blocks[-1].parameters():
                param.requires_grad = True
            for param in self.base_model.final_norm.parameters():
                param.requires_grad = True

        # Replace the output head with a classification head
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, num_classes),
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Get hidden states from base model (bypass output head)
        B, T = idx.shape
        tok_emb = self.base_model.tok_emb(idx)
        pos_emb = self.base_model.pos_emb(torch.arange(T, device=idx.device))
        x = self.base_model.drop_emb(tok_emb + pos_emb)

        for block in self.base_model.trf_blocks:
            x = block(x)
        x = self.base_model.final_norm(x)

        # Use the last token's hidden state for classification
        last_hidden = x[:, -1, :]  # (B, emb_dim)
        logits = self.classifier(last_hidden)  # (B, num_classes)
        return logits

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Chapter 6: Classification fine-tuning loop
# ──────────────────────────────────────────────

def finetune_classifier(
    classifier: GPTClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 1e-4,
) -> dict:
    """Fine-tune the classifier head."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    history = {"train_losses": [], "val_accs": []}

    classifier.to(device)
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        for inp, labels in train_loader:
            inp, labels = inp.to(device), labels.to(device)
            logits = classifier(inp)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_losses"].append(avg_loss)

        # Validation accuracy
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inp, labels in val_loader:
                inp, labels = inp.to(device), labels.to(device)
                preds = classifier(inp).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / max(total, 1)
        history["val_accs"].append(acc)
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss {avg_loss:.4f} | Val Acc {acc:.2%}")

    return history


# ──────────────────────────────────────────────
# Chapter 7: Supervised Fine-Tuning (SFT)
# ──────────────────────────────────────────────

def calc_sft_loss_batch(model, input_batch, target_batch, loss_mask, device):
    """
    Cross-entropy loss with loss masking.

    Only tokens where loss_mask == 1 contribute to the loss.
    This ensures the model learns to generate the *response* only,
    not memorize the prompt/instruction.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss_mask = loss_mask.to(device)

    logits = model(input_batch)  # (B, T, V)

    # Flatten
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = target_batch.view(B * T)
    mask_flat = loss_mask.view(B * T)

    # Per-token loss
    per_token_loss = nn.functional.cross_entropy(
        logits_flat, targets_flat, reduction="none"
    )

    # Apply mask
    masked_loss = (per_token_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    return masked_loss


def sft_finetune(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: BPETokenizer,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    checkpoint_dir: str | None = None,
    log_interval: int = 20,
    sample_prompt: str = "حل المسألة الرياضية التالية خطوة بخطوة.\nإذا كان لدى أحمد ٥ تفاحات وأعطى ٢ لصديقه، كم تفاحة تبقى لديه؟",
) -> dict:
    """
    Supervised Fine-Tuning with loss masking.

    The model learns to follow instructions by only computing loss
    on the response tokens, not on the instruction/prompt tokens.
    """
    checkpoint_dir = Path(checkpoint_dir or PROJECT_ROOT / "checkpoints" / "finetuned")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    total_steps = num_epochs * len(train_loader)
    history = {
        "train_losses": [],
        "val_losses": [],
        "samples": [],
        "steps": [],
    }

    model.to(device)
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  SFT Fine-Tuning — {model.count_parameters():,} parameters")
    print(f"  Device: {device}  |  Epochs: {num_epochs}  |  Steps: {total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inp, tgt, mask) in enumerate(train_loader):
            loss = calc_sft_loss_batch(model, inp, tgt, mask, device)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                history["train_losses"].append(loss.item())
                history["steps"].append(global_step)
                print(f"  Epoch {epoch+1}/{num_epochs} | "
                      f"Step {global_step}/{total_steps} | "
                      f"SFT Loss {loss.item():.4f} | "
                      f"Time {elapsed:.0f}s")

        # End-of-epoch validation
        model.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for inp, tgt, mask in val_loader:
                val_loss = calc_sft_loss_batch(model, inp, tgt, mask, device)
                val_loss_total += val_loss.item()
                val_batches += 1
        avg_val_loss = val_loss_total / max(val_batches, 1)
        history["val_losses"].append(avg_val_loss)
        print(f"  → Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        # Sample generation
        sample_ids = tokenizer.encode(sample_prompt)
        sample_tensor = torch.tensor([sample_ids], device=device)
        gen_ids = generate(
            model, sample_tensor,
            max_new_tokens=80,
            context_length=model.cfg["context_length"],
            temperature=0.7,
            top_k=25,
        )
        gen_text = tokenizer.decode(gen_ids[0].tolist())
        history["samples"].append({"epoch": epoch + 1, "text": gen_text})
        print(f"  Sample: {gen_text[:150]}...")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.cfg,
                "epoch": epoch,
                "val_loss": avg_val_loss,
            }, checkpoint_dir / "best_sft_model.pt")
            print(f"  ✓ Best SFT model saved (val_loss={avg_val_loss:.4f})")

        model.train()

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.cfg,
        "epoch": num_epochs,
        "val_loss": best_val_loss,
    }, checkpoint_dir / "final_sft_model.pt")

    with open(checkpoint_dir / "sft_history.json", "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\nSFT complete in {total_time:.0f}s  |  Best val loss: {best_val_loss:.4f}")
    return history


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def run_sft(
    pretrained_checkpoint: str | None = None,
    sft_data_path: str | None = None,
    tokenizer_path: str | None = None,
    num_epochs: int = 5,
    batch_size: int = 4,
    lr: float = 2e-5,
    device: str | None = None,
):
    """End-to-end SFT: load pretrained model → fine-tune → save."""
    pretrained_checkpoint = pretrained_checkpoint or str(
        PROJECT_ROOT / "checkpoints/pretrained/best_model.pt"
    )
    sft_data_path = sft_data_path or str(PROJECT_ROOT / "data/finetune/sft_data.json")
    tokenizer_path = tokenizer_path or str(PROJECT_ROOT / "data/tokenizer.json")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device)

    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)

    # Load pretrained model
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = GPTModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded pretrained model from {pretrained_checkpoint}")
    print(f"  Pretrain val_loss: {checkpoint.get('val_loss', 'N/A')}")

    # Build SFT dataset
    dataset = SFTDataset(sft_data_path, tokenizer, cfg["context_length"])
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"SFT Dataset: {len(dataset)} samples  (train={train_size}, val={val_size})")

    history = sft_finetune(
        model, train_loader, val_loader, tokenizer,
        device=device, num_epochs=num_epochs, lr=lr,
    )
    return model, tokenizer, history


if __name__ == "__main__":
    run_sft()
