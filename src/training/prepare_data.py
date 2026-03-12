"""
Data Preparation for Arabic Math Reasoning LLM.

Following Chapters 1-2 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

Data source: Omartificial-Intelligence-Space/Arabic-gsm8k-v2  (HuggingFace)

Produces:
  1. data/pretrain/data.txt          — raw Arabic text for next-token pretraining
  2. data/finetune/sft_data.json     — instruction–response pairs for SFT
  3. Trained BPE tokenizer saved to  data/tokenizer.json
  4. PyTorch Dataset & DataLoader wrappers
"""

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.bpe_tokenizer import BPETokenizer, normalize_arabic


# ──────────────────────────────────────────────
# 1.  Download & prepare raw data
# ──────────────────────────────────────────────

def download_arabic_gsm8k(data_dir: str | None = None) -> dict:
    """
    Download Arabic-gsm8k-v2 from HuggingFace and split into
    pretraining text and SFT pairs.
    """
    from datasets import load_dataset

    data_dir = Path(data_dir or PROJECT_ROOT / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Omartificial-Intelligence-Space/Arabic-gsm8k-v2 ...")
    ds = load_dataset("Omartificial-Intelligence-Space/Arabic-gsm8k-v2")

    # Collect all examples
    all_examples = []
    for split in ds:
        for row in ds[split]:
            question = row.get("question", row.get("instruction", ""))
            answer = row.get("answer", row.get("output", ""))
            if question and answer:
                all_examples.append({"question": question, "answer": answer})

    print(f"  Total examples collected: {len(all_examples)}")

    # ── Phase 1: Pretraining text ─────────────
    pretrain_dir = data_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = pretrain_dir / "data.txt"

    lines = []
    for ex in all_examples:
        lines.append(normalize_arabic(ex["question"]))
        lines.append(normalize_arabic(ex["answer"]))
        lines.append("")  # blank separator

    with open(pretrain_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Pretraining data: {pretrain_path}  "
          f"({pretrain_path.stat().st_size / 1024:.1f} KB, {len(lines)} lines)")

    # ── Phase 2: SFT instruction pairs ────────
    finetune_dir = data_dir / "finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    sft_path = finetune_dir / "sft_data.json"

    sft_records = []
    for ex in all_examples:
        sft_records.append({
            "instruction": "حل المسألة الرياضية التالية خطوة بخطوة.",
            "input": ex["question"],
            "output": ex["answer"],
        })

    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_records, f, ensure_ascii=False, indent=2)

    print(f"  SFT data: {sft_path}  ({len(sft_records)} pairs)")

    return {
        "pretrain_path": str(pretrain_path),
        "sft_path": str(sft_path),
        "num_examples": len(all_examples),
    }


# ──────────────────────────────────────────────
# 2.  Train BPE tokenizer on the corpus
# ──────────────────────────────────────────────

def train_tokenizer(
    text_path: str,
    vocab_size: int = 5000,
    save_path: str | None = None,
) -> BPETokenizer:
    """Train a BPE tokenizer on the pretraining corpus."""
    text_path = Path(text_path)
    save_path = save_path or str(PROJECT_ROOT / "data" / "tokenizer.json")

    print(f"Training BPE tokenizer (target vocab={vocab_size}) ...")
    text = text_path.read_text(encoding="utf-8")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text, verbose=True)
    tokenizer.save(save_path)

    print(f"  Tokenizer saved to {save_path}  (actual vocab={tokenizer.vocab_size})")
    return tokenizer


# ──────────────────────────────────────────────
# 3.  PyTorch Datasets
# ──────────────────────────────────────────────

class PretrainDataset(Dataset):
    """
    Dataset for causal language modeling (next-token prediction).

    Each sample is a window of (context_length + 1) tokens:
        input  = tokens[:-1]
        target = tokens[1:]
    """

    def __init__(
        self,
        text_path: str,
        tokenizer: BPETokenizer,
        context_length: int = 256,
        stride: int = 128,
    ):
        text = Path(text_path).read_text(encoding="utf-8")
        self.token_ids = tokenizer.encode(text)
        self.context_length = context_length

        # Sliding window with stride
        self.samples = []
        for i in range(0, len(self.token_ids) - context_length, stride):
            chunk = self.token_ids[i : i + context_length + 1]
            self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return chunk[:-1], chunk[1:]  # input, target


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (instruction following).

    Each sample is formatted as:
        <|im_start|> instruction \n input <|im_end|>
        <|im_start|> output <|im_end|>

    Loss is only computed on the output tokens (loss masking).
    """

    PROMPT_TEMPLATE = "{instruction}\n{input}"

    def __init__(
        self,
        sft_path: str,
        tokenizer: BPETokenizer,
        context_length: int = 256,
    ):
        with open(sft_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.samples = []
        self.loss_masks = []
        self.tokenizer = tokenizer
        self.context_length = context_length

        im_start = tokenizer.vocab.get("<|im_start|>", tokenizer.bos_id)
        im_end = tokenizer.vocab.get("<|im_end|>", tokenizer.eos_id)

        for rec in records:
            prompt = self.PROMPT_TEMPLATE.format(
                instruction=rec["instruction"],
                input=rec.get("input", ""),
            )
            prompt_ids = [im_start] + tokenizer.encode(prompt) + [im_end]
            response_ids = [im_start] + tokenizer.encode(rec["output"]) + [im_end]

            full_ids = prompt_ids + response_ids

            # Truncate or pad to context_length + 1
            if len(full_ids) > context_length + 1:
                full_ids = full_ids[: context_length + 1]
                # Rebuild mask
                prompt_len = min(len(prompt_ids), context_length + 1)
            else:
                prompt_len = len(prompt_ids)
                pad_len = (context_length + 1) - len(full_ids)
                full_ids = full_ids + [tokenizer.pad_id] * pad_len

            # Loss mask: 0 for prompt tokens + padding, 1 for response tokens
            mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
            # Zero-out padding
            for i in range(len(full_ids)):
                if full_ids[i] == tokenizer.pad_id:
                    mask[i] = 0

            self.samples.append(torch.tensor(full_ids, dtype=torch.long))
            self.loss_masks.append(torch.tensor(mask[: context_length + 1], dtype=torch.float))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        mask = self.loss_masks[idx]
        return chunk[:-1], chunk[1:], mask[1:]  # input, target, loss_mask


# ──────────────────────────────────────────────
# 4.  DataLoader factory
# ──────────────────────────────────────────────

def create_pretrain_dataloader(
    text_path: str,
    tokenizer: BPETokenizer,
    context_length: int = 256,
    batch_size: int = 8,
    stride: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PretrainDataset(text_path, tokenizer, context_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def create_sft_dataloader(
    sft_path: str,
    tokenizer: BPETokenizer,
    context_length: int = 256,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = SFTDataset(sft_path, tokenizer, context_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    info = download_arabic_gsm8k()
    tokenizer = train_tokenizer(info["pretrain_path"], vocab_size=5000)

    # Quick sanity check
    sample = "كم عدد التفاحات المتبقية إذا كان لديك ١٠ وأعطيت ٣؟"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"\nSanity check:")
    print(f"  Original : {sample}")
    print(f"  Encoded  : {encoded[:20]}... ({len(encoded)} tokens)")
    print(f"  Decoded  : {decoded}")
