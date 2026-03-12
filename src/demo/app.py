"""
Interactive Gradio Demo — Arabic Math Reasoning LLM.

Visualizes every step of the LLM pipeline following the book:

  Tab 1 │ Chapter 1: Understanding LLMs      — overview, data stats, scaling
  Tab 2 │ Chapter 2: Working With Text Data   — tokenization, vocab, encoding/decoding
  Tab 3 │ Chapter 3: Attention Mechanisms     — QKV math, causal mask, attention heatmap
  Tab 4 │ Chapter 4: GPT Architecture         — model diagram, layer info, embeddings
  Tab 5 │ Chapter 5: Pretraining              — loss curves, sample generations
  Tab 6 │ Chapter 6: Fine-Tuning (Classify)   — transfer learning, frozen vs full
  Tab 7 │ Chapter 7: Instruction SFT          — prompt formatting, loss masking, generation
  Tab 8 │ Evaluation & Error Analysis         — perplexity, accuracy, error categories
"""

import sys
import json
import os
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.bpe_tokenizer import BPETokenizer, normalize_arabic, pre_tokenize
from src.model.transformer import GPTModel, GPT_CONFIG, generate, generate_step_by_step
from src.model.attention import scaled_dot_product_attention


# ──────────────────────────────────────────────
# Globals (loaded on startup)
# ──────────────────────────────────────────────

TOKENIZER: BPETokenizer | None = None
PRETRAINED_MODEL: GPTModel | None = None
SFT_MODEL: GPTModel | None = None
DEVICE = torch.device("cpu")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_assets():
    """Load tokenizer and models if checkpoints exist."""
    global TOKENIZER, PRETRAINED_MODEL, SFT_MODEL, DEVICE
    DEVICE = get_device()

    tok_path = PROJECT_ROOT / "data" / "tokenizer.json"
    if tok_path.exists():
        TOKENIZER = BPETokenizer.load(str(tok_path))
        print(f"Loaded tokenizer (vocab={TOKENIZER.vocab_size})")

    pt_path = PROJECT_ROOT / "checkpoints" / "pretrained" / "best_model.pt"
    if not pt_path.exists():
        pt_path = PROJECT_ROOT / "checkpoints" / "pretrained" / "final_model.pt"
    if pt_path.exists():
        ckpt = torch.load(pt_path, map_location=DEVICE, weights_only=False)
        cfg = ckpt["config"]
        PRETRAINED_MODEL = GPTModel(cfg)
        PRETRAINED_MODEL.load_state_dict(ckpt["model_state_dict"])
        PRETRAINED_MODEL.to(DEVICE).eval()
        print(f"Loaded pretrained model ({PRETRAINED_MODEL.count_parameters():,} params)")

    sft_path = PROJECT_ROOT / "checkpoints" / "finetuned" / "best_sft_model.pt"
    if not sft_path.exists():
        sft_path = PROJECT_ROOT / "checkpoints" / "finetuned" / "final_sft_model.pt"
    if sft_path.exists():
        ckpt = torch.load(sft_path, map_location=DEVICE, weights_only=False)
        cfg = ckpt["config"]
        SFT_MODEL = GPTModel(cfg)
        SFT_MODEL.load_state_dict(ckpt["model_state_dict"])
        SFT_MODEL.to(DEVICE).eval()
        print(f"Loaded SFT model ({SFT_MODEL.count_parameters():,} params)")


# ──────────────────────────────────────────────
# Tab 1: Understanding LLMs
# ──────────────────────────────────────────────

def tab1_overview():
    """Return project overview markdown."""
    # Load data stats if available
    sft_path = PROJECT_ROOT / "data" / "finetune" / "sft_data.json"
    pretrain_path = PROJECT_ROOT / "data" / "pretrain" / "data.txt"

    stats = "### Dataset Statistics\n"
    if pretrain_path.exists():
        text = pretrain_path.read_text(encoding="utf-8")
        stats += f"- **Pretraining text**: {len(text):,} characters, {len(text.split()):,} words, {len(text.splitlines()):,} lines\n"
    if sft_path.exists():
        with open(sft_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats += f"- **SFT instruction pairs**: {len(data):,}\n"
    if TOKENIZER:
        stats += f"- **Vocabulary size**: {TOKENIZER.vocab_size:,} tokens\n"
    if PRETRAINED_MODEL:
        stats += f"- **Model parameters**: {PRETRAINED_MODEL.count_parameters():,}\n"

    overview = f"""
# Arabic Math Reasoning LLM — From Scratch

## Project Overview
This project implements a **GPT (Decoder-Only Transformer)** entirely from scratch
using PyTorch, following *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

**Task**: Arabic mathematical reasoning using the **Arabic-gsm8k-v2** dataset.

## How LLMs Work — Key Concepts
1. **Next-Token Prediction**: The model learns to predict the next token given all previous tokens
2. **Tokens & Embeddings**: Text is split into sub-word tokens, each mapped to a dense vector
3. **Transformer Architecture**: Self-attention allows each token to attend to all previous tokens
4. **Scaling Laws**: More data + more parameters → better performance (Kaplan et al., 2020)

## Pipeline
```
Raw Arabic Text → BPE Tokenization → Token IDs → Embeddings
→ Transformer Blocks (Attention + FFN + Residuals + LayerNorm)
→ Output Logits → Next Token Prediction → Generated Text
```

{stats}

## Book Chapters Covered
| Chapter | Topic | Status |
|---------|-------|--------|
| 1 | Understanding Large Language Models | ✅ |
| 2 | Working With Text Data | ✅ |
| 3 | Coding Attention Mechanisms | ✅ |
| 4 | Implementing a GPT Model | ✅ |
| 5 | Pretraining on Unlabeled Data | ✅ |
| 6 | Fine-Tuning for Classification | ✅ |
| 7 | Fine-Tuning to Follow Instructions | ✅ |
"""
    return overview


# ──────────────────────────────────────────────
# Tab 2: Tokenization
# ──────────────────────────────────────────────

def tab2_tokenize(text: str):
    """Tokenize input text and show detailed breakdown."""
    if not TOKENIZER:
        return "Tokenizer not loaded. Run data preparation first.", "", ""

    # Normalize
    normalized = normalize_arabic(text)

    # Pre-tokenize (word-level)
    words = pre_tokenize(normalized)

    # BPE encode
    token_ids = TOKENIZER.encode(text)
    details = TOKENIZER.get_token_details(text)

    # Decode back
    decoded = TOKENIZER.decode(token_ids)

    # Build visualization
    viz_lines = ["### Step-by-Step Tokenization\n"]
    viz_lines.append(f"**Original text**: {text}\n")
    viz_lines.append(f"**Normalized**: {normalized}\n")
    viz_lines.append(f"**Pre-tokenized words** ({len(words)}): {' | '.join(words)}\n")
    viz_lines.append(f"**BPE tokens** ({len(token_ids)}):\n")

    token_table = "| Token | ID | Source Word |\n|---|---|---|\n"
    for d in details:
        token_table += f"| `{d['token']}` | {d['id']} | {d['source_word']} |\n"

    viz_lines.append(token_table)
    viz_lines.append(f"\n**Decoded back**: {decoded}")
    viz_lines.append(f"\n**Compression ratio**: {len(text)} chars → {len(token_ids)} tokens "
                     f"({len(text)/max(len(token_ids),1):.1f} chars/token)")

    # Vocab stats
    vocab_info = f"""### Vocabulary Statistics
- Total vocabulary: {TOKENIZER.vocab_size:,} tokens
- Special tokens: {len(TOKENIZER.SPECIAL_TOKENS)}
- BPE merges learned: {len(TOKENIZER.merges):,}
"""

    return "\n".join(viz_lines), vocab_info, str(token_ids)


# ──────────────────────────────────────────────
# Tab 3: Attention Visualization
# ──────────────────────────────────────────────

def tab3_attention(text: str, layer_idx: int = 0, head_idx: int = 0):
    """Visualize attention patterns for input text."""
    if not TOKENIZER or not PRETRAINED_MODEL:
        return "Model not loaded.", None

    model = SFT_MODEL if SFT_MODEL else PRETRAINED_MODEL

    ids = TOKENIZER.encode(text)
    if len(ids) > model.cfg["context_length"]:
        ids = ids[:model.cfg["context_length"]]

    inp = torch.tensor([ids], device=DEVICE)

    with torch.no_grad():
        logits, attn_maps = model(inp, return_attention=True)

    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[1]

    layer_idx = min(layer_idx, n_layers - 1)
    head_idx = min(head_idx, n_heads - 1)

    attn = attn_maps[layer_idx][0, head_idx].cpu().numpy()

    # Build text labels
    details = TOKENIZER.get_token_details(text)
    labels = [d["token"][:8] for d in details][:attn.shape[0]]

    # Create matplotlib figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Attention heatmap
    ax = axes[0]
    im = ax.imshow(attn[:len(labels), :len(labels)], cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Attention Weights — Layer {layer_idx}, Head {head_idx}", fontsize=12)
    ax.set_xlabel("Key positions (attended to)")
    ax.set_ylabel("Query positions (attending from)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Causal mask visualization
    ax2 = axes[1]
    T = len(labels)
    causal_mask = np.tril(np.ones((T, T)))
    ax2.imshow(causal_mask, cmap="Greys", aspect="auto")
    ax2.set_title("Causal Mask (lower triangular)", fontsize=12)
    ax2.set_xlabel("Key positions")
    ax2.set_ylabel("Query positions")

    plt.tight_layout()

    # QKV explanation
    explanation = f"""### Attention Mechanism — Chapter 3

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**What you see:**
- **Left**: Attention weights for Layer {layer_idx}, Head {head_idx}
  - Brighter = token pays more attention to that position
  - Row = query token, Column = key token it attends to
- **Right**: Causal mask ensures tokens only attend to past positions

**Architecture details:**
- d_model = {model.cfg['emb_dim']}
- n_heads = {model.cfg['n_heads']}
- d_k = {model.cfg['emb_dim'] // model.cfg['n_heads']} per head
- {n_layers} transformer layers

**Input**: {len(ids)} tokens
**QKV shapes**: ({len(ids)}, {model.cfg['emb_dim'] // model.cfg['n_heads']}) per head
"""
    return explanation, fig


# ──────────────────────────────────────────────
# Tab 4: Model Architecture
# ──────────────────────────────────────────────

def tab4_architecture():
    """Show model architecture details."""
    model = SFT_MODEL if SFT_MODEL else PRETRAINED_MODEL

    if not model:
        cfg = GPT_CONFIG
        arch_info = "**Model not yet trained.** Showing default configuration.\n\n"
    else:
        cfg = model.cfg
        arch_info = f"**Trained model loaded** — {model.count_parameters():,} parameters\n\n"

    # Layer info table
    if model:
        layers = model.get_layer_info()
        layer_table = "| Layer | Shape | Parameters |\n|---|---|---|\n"
        total = 0
        for l in layers:
            layer_table += f"| {l['name']} | {l['shape']} | {l['params']:,} |\n"
            total += l['params']
        layer_table += f"| **Total** | — | **{total:,}** |\n"
    else:
        layer_table = ""

    arch_md = f"""### GPT Model Architecture — Chapter 4

{arch_info}

#### Configuration
```python
{{
    "vocab_size": {cfg.get('vocab_size', 5000)},
    "context_length": {cfg.get('context_length', 256)},
    "emb_dim": {cfg.get('emb_dim', 512)},
    "n_heads": {cfg.get('n_heads', 8)},
    "n_layers": {cfg.get('n_layers', 6)},
    "drop_rate": {cfg.get('drop_rate', 0.1)},
}}
```

#### Architecture Diagram
```
Input Token IDs  (B, T)
        │
        ▼
┌─────────────────────────┐
│   Token Embedding       │  vocab_size × emb_dim
│   + Positional Embedding│  context_length × emb_dim
│   + Dropout             │
└────────────┬────────────┘
             │
     ┌───────▼───────┐
     │  Transformer  │ × {cfg.get('n_layers', 6)} layers
     │    Block      │
     │ ┌───────────┐ │
     │ │ LayerNorm │ │
     │ │ Multi-Head│ │  {cfg.get('n_heads', 8)} heads, d_k={cfg.get('emb_dim', 512)//cfg.get('n_heads', 8)}
     │ │ Attention │ │
     │ │ + Residual│ │
     │ ├───────────┤ │
     │ │ LayerNorm │ │
     │ │ FFN (GELU)│ │  {cfg.get('emb_dim', 512)} → {4*cfg.get('emb_dim', 512)} → {cfg.get('emb_dim', 512)}
     │ │ + Residual│ │
     │ └───────────┘ │
     └───────┬───────┘
             │
     ┌───────▼───────┐
     │  Final        │
     │  LayerNorm    │
     └───────┬───────┘
             │
     ┌───────▼───────┐
     │  Output Head  │  emb_dim → vocab_size
     │  (weight-tied)│
     └───────┬───────┘
             │
             ▼
    Logits  (B, T, vocab_size)
```

#### Layer-by-Layer Breakdown
{layer_table}

#### Key Design Decisions
- **Pre-norm**: LayerNorm before attention/FFN (more stable training)
- **Weight tying**: Output head shares weights with token embeddings
- **GELU activation**: Smoother than ReLU, used in GPT-2/3
- **Causal masking**: Ensures autoregressive property (no future peeking)
"""
    return arch_md


# ──────────────────────────────────────────────
# Tab 5: Pretraining
# ──────────────────────────────────────────────

def tab5_pretraining():
    """Show pretraining results and loss curves."""
    history_path = PROJECT_ROOT / "checkpoints" / "pretrained" / "pretrain_history.json"

    if not history_path.exists():
        return "No pretraining history found. Run pretraining first.", None

    with open(history_path, "r") as f:
        history = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    if history.get("train_losses"):
        steps = history.get("steps", list(range(len(history["train_losses"]))))
        axes[0].plot(steps[:len(history["train_losses"])],
                     history["train_losses"], "b-", alpha=0.7, label="Train Loss")
        if history.get("val_losses"):
            val_steps = [steps[min(i * (len(steps)//max(len(history["val_losses"]),1)),
                                  len(steps)-1)]
                         for i in range(len(history["val_losses"]))]
            axes[0].plot(val_steps, history["val_losses"],
                         "r-o", markersize=4, label="Val Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss — Chapter 5")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Learning rate schedule
    if history.get("lrs"):
        axes[1].plot(steps[:len(history["lrs"])],
                     history["lrs"], "g-", alpha=0.7)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Cosine LR Schedule with Warmup")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Samples during training
    samples_md = "### Generated Samples During Training\n\n"
    if history.get("samples"):
        for s in history["samples"]:
            samples_md += f"**Step {s['step']}**: {s['text'][:200]}\n\n---\n\n"
    else:
        samples_md += "No samples recorded.\n"

    return samples_md, fig


# ──────────────────────────────────────────────
# Tab 6 & 7: Fine-Tuning & Generation
# ──────────────────────────────────────────────

def tab67_finetune_and_generate(question: str, temperature: float, top_k: int, max_tokens: int):
    """Compare pretrained vs SFT model on the same question."""
    if not TOKENIZER:
        return "Tokenizer not loaded.", "", ""

    prompt = f"حل المسألة الرياضية التالية خطوة بخطوة.\n{question}"

    results = {}
    for name, model in [("pretrained", PRETRAINED_MODEL), ("sft", SFT_MODEL)]:
        if model is None:
            results[name] = "Model not loaded."
            continue

        ids = TOKENIZER.encode(prompt)
        inp = torch.tensor([ids], device=DEVICE)

        gen_ids = generate(
            model, inp,
            max_new_tokens=int(max_tokens),
            context_length=model.cfg["context_length"],
            temperature=temperature,
            top_k=int(top_k),
        )
        results[name] = TOKENIZER.decode(gen_ids[0].tolist())

    # Step-by-step visualization
    step_viz = ""
    active_model = SFT_MODEL if SFT_MODEL else PRETRAINED_MODEL
    if active_model:
        ids = TOKENIZER.encode(prompt)
        inp = torch.tensor([ids], device=DEVICE)
        steps = generate_step_by_step(
            active_model, inp,
            max_new_tokens=min(10, int(max_tokens)),
            context_length=active_model.cfg["context_length"],
            temperature=temperature,
            top_k=int(top_k),
        )
        step_viz = "### Token-by-Token Generation (first 10 tokens)\n\n"
        step_viz += "| Step | Token | ID | Prob | Top-5 Candidates |\n"
        step_viz += "|------|-------|----|------|------------------|\n"
        for s in steps:
            tok = TOKENIZER.inverse_vocab.get(s["token_id"], "?")
            top5 = ", ".join(
                f"{TOKENIZER.inverse_vocab.get(tid, '?')} ({p:.2f})"
                for tid, p in zip(s["top5_ids"], s["top5_probs"])
            )
            step_viz += f"| {s['step']} | `{tok}` | {s['token_id']} | {s['probability']:.3f} | {top5} |\n"

    comparison = f"""### Generation Comparison

#### Pretrained Model (Chapter 5)
> {results.get('pretrained', 'Not available')}

---

#### SFT Model (Chapter 7)
> {results.get('sft', 'Not available')}

---

#### Explanation
- **Pretrained model**: Learned language patterns from raw text (next-token prediction)
- **SFT model**: Fine-tuned to follow instructions with **loss masking** —
  loss is only computed on response tokens, not on the instruction prompt

#### Prompt Format (Chapter 7)
```
<|im_start|> حل المسألة الرياضية التالية خطوة بخطوة.
[question] <|im_end|>
<|im_start|> [model generates response here] <|im_end|>
```
"""

    return comparison, step_viz, results.get("sft", results.get("pretrained", ""))


# ──────────────────────────────────────────────
# Tab 8: Evaluation
# ──────────────────────────────────────────────

def tab8_evaluation():
    """Show evaluation results."""
    results_path = PROJECT_ROOT / "results" / "evaluation_results.json"

    if not results_path.exists():
        return "No evaluation results found. Run evaluation first.", None

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Error analysis pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if "error_analysis" in results:
        cats = results["error_analysis"].get("category_counts", {})
        if cats:
            labels = list(cats.keys())
            sizes = list(cats.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                        startangle=90)
            axes[0].set_title("Error Analysis — Failure Categories")

    # Quality stats
    if "generation_quality" in results:
        stats = results["generation_quality"].get("stats", {})
        if stats:
            metrics = list(stats.keys())
            values = list(stats.values())
            bars = axes[1].barh(metrics, values, color="steelblue")
            axes[1].set_title("Generation Quality Metrics")
            for bar, val in zip(bars, values):
                axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                             f"{val:.2f}", va="center")

    plt.tight_layout()

    eval_md = f"""### Evaluation Results

| Metric | Value |
|--------|-------|
| **Perplexity** | {results.get('perplexity', 'N/A')} |
| **Answer Accuracy** | {results.get('answer_accuracy', {}).get('accuracy', 'N/A')} |
| **Correct** | {results.get('answer_accuracy', {}).get('correct', 'N/A')} / {results.get('answer_accuracy', {}).get('total', 'N/A')} |

### Error Categories
"""
    if "error_analysis" in results:
        cats = results["error_analysis"].get("category_counts", {})
        for cat, count in cats.items():
            eval_md += f"- **{cat}**: {count}\n"

    return eval_md, fig


# ──────────────────────────────────────────────
# Build Gradio App
# ──────────────────────────────────────────────

def build_app():
    import gradio as gr

    load_assets()

    with gr.Blocks(
        title="Arabic Math LLM — From Scratch",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .rtl { direction: rtl; text-align: right; }
        .main-title { text-align: center; margin-bottom: 20px; }
        """,
    ) as app:
        gr.Markdown(
            "# Arabic Math Reasoning LLM — Built From Scratch\n"
            "### Following *Build a Large Language Model (From Scratch)* by Sebastian Raschka\n"
            "**Dataset**: Omartificial-Intelligence-Space/Arabic-gsm8k-v2",
            elem_classes="main-title",
        )

        # ── Tab 1: Overview ──
        with gr.Tab("Ch.1 Understanding LLMs"):
            overview_md = gr.Markdown(tab1_overview())

        # ── Tab 2: Tokenization ──
        with gr.Tab("Ch.2 Tokenization"):
            gr.Markdown("### Interactive Arabic BPE Tokenization")
            with gr.Row():
                tok_input = gr.Textbox(
                    label="Arabic text",
                    value="كم عدد التفاحات المتبقية إذا كان لديك ١٠ وأعطيت ٣؟",
                    lines=2, rtl=True,
                )
            tok_btn = gr.Button("Tokenize", variant="primary")
            tok_result = gr.Markdown()
            tok_stats = gr.Markdown()
            tok_ids = gr.Textbox(label="Token IDs", interactive=False)
            tok_btn.click(tab2_tokenize, tok_input, [tok_result, tok_stats, tok_ids])

        # ── Tab 3: Attention ──
        with gr.Tab("Ch.3 Attention"):
            gr.Markdown("### Attention Mechanism Visualization")
            with gr.Row():
                attn_input = gr.Textbox(
                    label="Arabic text",
                    value="اذا كان لدى احمد خمسه تفاحات",
                    lines=2, rtl=True,
                )
                attn_layer = gr.Slider(0, 5, value=0, step=1, label="Layer")
                attn_head = gr.Slider(0, 7, value=0, step=1, label="Head")
            attn_btn = gr.Button("Visualize Attention", variant="primary")
            attn_explanation = gr.Markdown()
            attn_plot = gr.Plot()
            attn_btn.click(
                tab3_attention, [attn_input, attn_layer, attn_head],
                [attn_explanation, attn_plot],
            )

        # ── Tab 4: Architecture ──
        with gr.Tab("Ch.4 GPT Architecture"):
            arch_md = gr.Markdown(tab4_architecture())

        # ── Tab 5: Pretraining ──
        with gr.Tab("Ch.5 Pretraining"):
            pt_btn = gr.Button("Load Pretraining Results", variant="primary")
            pt_samples = gr.Markdown()
            pt_plot = gr.Plot()
            pt_btn.click(tab5_pretraining, [], [pt_samples, pt_plot])

        # ── Tab 6-7: Fine-Tuning & Generation ──
        with gr.Tab("Ch.6-7 Fine-Tuning & Generation"):
            gr.Markdown("### Compare Pretrained vs Instruction-Tuned Model")
            with gr.Row():
                gen_question = gr.Textbox(
                    label="Math question (Arabic)",
                    value="إذا كان لدى سارة ١٢ كتاباً وأعطت ٤ لصديقتها، كم كتاباً تبقى لديها؟",
                    lines=3, rtl=True,
                )
            with gr.Row():
                gen_temp = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                gen_topk = gr.Slider(1, 100, value=25, step=1, label="Top-K")
                gen_maxlen = gr.Slider(10, 200, value=100, step=10, label="Max Tokens")
            gen_btn = gr.Button("Generate & Compare", variant="primary")
            gen_comparison = gr.Markdown()
            gen_steps = gr.Markdown()
            gen_raw = gr.Textbox(label="Raw output", interactive=False, rtl=True)
            gen_btn.click(
                tab67_finetune_and_generate,
                [gen_question, gen_temp, gen_topk, gen_maxlen],
                [gen_comparison, gen_steps, gen_raw],
            )

        # ── Tab 8: Evaluation ──
        with gr.Tab("Ch.8 Evaluation"):
            eval_btn = gr.Button("Load Evaluation Results", variant="primary")
            eval_md = gr.Markdown()
            eval_plot = gr.Plot()
            eval_btn.click(tab8_evaluation, [], [eval_md, eval_plot])

    return app


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
