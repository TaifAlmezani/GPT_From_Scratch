# Arabic Math Reasoning LLM — Built From Scratch

Implementation of a **GPT** built entirely from scratch using PyTorch, following *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The model is trained on the [Arabic-gsm8k-v2](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k-v2) dataset for **Arabic mathematical reasoning**.

---


## Architecture

```
GPT Model (~30M parameters)
├── Token Embedding       (vocab_size × 512)
├── Positional Embedding  (256 × 512)
├── Dropout
├── 6 × Transformer Block
│   ├── LayerNorm → Multi-Head Attention (8 heads, d_k=64) → Residual
│   └── LayerNorm → Feed-Forward (512→2048→512, GELU) → Residual
├── Final LayerNorm
└── Output Head (weight-tied with Token Embedding)
```

### Key Design Choices
- **Pre-norm architecture** (LayerNorm before attention/FFN) for training stability
- **Weight tying** between token embedding and output head (reduces parameters)
- **GELU activation** (smoother than ReLU, used in GPT-2/3)
- **Causal masking** for autoregressive generation
- **Cosine LR schedule with warmup** for optimization
- **Loss masking in SFT** (loss only on response tokens, not instructions)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data & Train Tokenizer

```bash
python -m src.training.prepare_data
```


### 3. Pretrain the Model

```bash
python -m src.training.pretrain
```

### 4. Fine-Tune with Instructions (SFT)

```bash
python -m src.training.finetune
```

### 5. Run the Interactive Demo

```bash
python -m src.demo.app
```


### 6. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/02_full_pipeline_demo.ipynb
```

> The Colab-style notebook `notebooks/Arabic_Math_LLM_Colab.ipynb` now auto-creates folders and saves:
> - `data/pretrain/data.txt`
> - `data/finetune/sft_data.json`
> - `data/tokenizer.json`
> - `checkpoints/pretrained/best_pretrained.pt`
> - `checkpoints/pretrained/final_pretrained.pt`
> - `checkpoints/finetuned/best_sft.pt`
> - `results/sample_generations/sft_samples.json`

---

## Project Structure

```
ArabicLLMFromScratch/
├── data/
│   ├── pretrain/
│   │   └── data.txt                    # Raw Arabic text for pretraining
│   ├── finetune/
│   │   └── sft_data.json               # Instruction-response pairs
│   └── tokenizer.json                  # Trained BPE tokenizer
├── src/
│   ├── tokenizer/
│   │   └── bpe_tokenizer.py            # BPE tokenizer from scratch (Ch.2)
│   ├── model/
│   │   ├── attention.py                # Attention mechanisms (Ch.3)
│   │   └── transformer.py              # GPT model + generation (Ch.4)
│   ├── training/
│   │   ├── prepare_data.py             # Data download + Dataset/DataLoader
│   │   ├── pretrain.py                 # Pretraining loop (Ch.5)
│   │   └── finetune.py                 # Classification + SFT (Ch.6-7)
│   ├── evaluation/
│   │   ├── metrics.py                  # Perplexity, accuracy, LLM-judge
│   │   └── error_analysis.py           # Error categorization
│   └── demo/
│       └── app.py                      # Gradio interactive demo
├── checkpoints/
│   ├── pretrained/                     # Pretrained model checkpoints
│   └── finetuned/                      # SFT model checkpoints
├── results/
│   ├── sample_generations/
│   └── plots/                          # Loss curves, attention maps, etc.
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Data analysis
│   └── 02_full_pipeline_demo.ipynb     # Complete step-by-step walkthrough
├── requirements.txt
└── README.md
```

---

## Data

### Pretraining Data (Phase 1)
- **Source**: Arabic-gsm8k-v2 (all questions and answers concatenated)
- **Format**: Raw UTF-8 text
- **Purpose**: Teach the model Arabic language patterns via next-token prediction
- **Processing**: Unicode normalization (NFKC), optional diacritic removal, alef normalization

### SFT Data (Phase 2)
- **Source**: Same dataset, formatted as instruction-response pairs
- **Format**: JSON with `instruction`, `input`, `output` fields
- **Purpose**: Teach the model to solve math problems step by step
- **Loss Masking**: Only response tokens contribute to the loss

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Perplexity** | exp(avg cross-entropy loss) — lower is better |
| **Answer Accuracy** | Exact/fuzzy match on extracted numeric answers |
| **LLM-as-Judge** | Heuristic scoring: length, number overlap, word overlap |
| **Error Analysis** | Categorization: repetition, wrong number, no answer, missing steps |

---

## Demo Features

The demo visualizes every step of the pipeline:

1. **Ch.1 Understanding LLMs** — Project overview, dataset stats
2. **Ch.2 Tokenization** — Interactive BPE tokenization with token breakdown
3. **Ch.3 Attention** — Attention weight heatmaps, causal mask visualization
4. **Ch.4 GPT Architecture** — Model diagram, layer-by-layer parameter breakdown
5. **Ch.5 Pretraining** — Training/validation loss curves, LR schedule
6. **Ch.6-7 Fine-Tuning** — Side-by-side comparison, token-by-token generation
7. **Ch.8 Evaluation** — Perplexity, accuracy, error category charts

## Demo Media (Pics + Video)

### Screenshots



### Demo Video



---

## Technical Limitations

1. **Model Size**: ~30M parameters — too small for complex multi-step reasoning
2. **Training Data**: ~8K examples from Arabic GSM8K — limited diversity
3. **Tokenizer**: Character-level BPE may not capture all Arabic morphology
4. **No RLHF**: Only supervised fine-tuning, no preference optimization
5. **Context Length**: 256 tokens — limits long problem solving

---

## Future Work

- Scale to 100M+ parameters with more Arabic pretraining data
- Implement RLHF / DPO for better alignment
- Chain-of-thought prompting for multi-step reasoning
- Multi-task fine-tuning (translation, summarization, QA)
- Flash Attention for efficient training on longer contexts
- Arabic-specific tokenizer improvements (morphological awareness)

---

## References

- Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. Manning Publications.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI.
- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv.
- Dataset: [Omartificial-Intelligence-Space/Arabic-gsm8k-v2](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k-v2)

####  *Done by Taif Almezani* 
