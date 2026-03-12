"""
Evaluation Metrics for Arabic Math Reasoning LLM.

Implements:
  1. Perplexity               — standard LM evaluation metric
  2. Answer Accuracy           — exact/fuzzy match on math answers
  3. LLM-as-Judge              — use a strong model to score quality (optional)
  4. Error Analysis             — categorize failure modes
  5. Generation Quality Report — coherence, repetition, length analysis
"""

import re
import sys
import json
import math
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.transformer import GPTModel, generate


# ──────────────────────────────────────────────
# 1. Perplexity
# ──────────────────────────────────────────────

def compute_perplexity(model: GPTModel, data_loader, device: torch.device) -> float:
    """
    Perplexity = exp(average cross-entropy loss).

    Lower is better.  A perplexity of P means the model is, on average,
    as uncertain as choosing uniformly among P tokens.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for inp, tgt in data_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += tgt.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# ──────────────────────────────────────────────
# 2. Answer accuracy (math reasoning)
# ──────────────────────────────────────────────

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def extract_number(text: str) -> float | None:
    """Extract the final numeric answer from Arabic math text."""
    text = text.translate(ARABIC_DIGITS)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def compute_answer_accuracy(
    model: GPTModel,
    tokenizer: BPETokenizer,
    questions: list[str],
    gold_answers: list[str],
    device: torch.device,
    max_new_tokens: int = 100,
) -> dict:
    """
    Generate answers for math questions and compare with gold answers.

    Returns accuracy and per-question results.
    """
    model.eval()
    results = []
    correct = 0

    for q, gold in zip(questions, gold_answers):
        prompt = f"حل المسألة الرياضية التالية خطوة بخطوة.\n{q}"
        ids = tokenizer.encode(prompt)
        inp = torch.tensor([ids], device=device)

        gen_ids = generate(
            model, inp,
            max_new_tokens=max_new_tokens,
            context_length=model.cfg["context_length"],
            temperature=0.3,
            top_k=10,
        )
        gen_text = tokenizer.decode(gen_ids[0].tolist())

        pred_num = extract_number(gen_text)
        gold_num = extract_number(gold)

        is_correct = (pred_num is not None and gold_num is not None
                      and abs(pred_num - gold_num) < 1e-6)
        if is_correct:
            correct += 1

        results.append({
            "question": q,
            "gold_answer": gold,
            "generated": gen_text,
            "pred_number": pred_num,
            "gold_number": gold_num,
            "correct": is_correct,
        })

    accuracy = correct / max(len(questions), 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(questions),
        "results": results,
    }


# ──────────────────────────────────────────────
# 3. LLM-as-Judge (placeholder / local scoring)
# ──────────────────────────────────────────────

def llm_judge_score(generated: str, reference: str) -> dict:
    """
    Score generated text quality against reference.

    This uses simple heuristic scoring. In production, you would call
    a strong external LLM (GPT-4, Claude) via API for proper judging.
    """
    scores = {}

    # Length ratio (penalize too short or too long)
    len_ratio = len(generated) / max(len(reference), 1)
    scores["length_score"] = max(0, 1 - abs(1 - len_ratio))

    # Digit overlap (important for math)
    gen_nums = set(re.findall(r"\d+", generated.translate(ARABIC_DIGITS)))
    ref_nums = set(re.findall(r"\d+", reference.translate(ARABIC_DIGITS)))
    if ref_nums:
        scores["number_overlap"] = len(gen_nums & ref_nums) / len(ref_nums)
    else:
        scores["number_overlap"] = 1.0

    # Word overlap (Jaccard similarity)
    gen_words = set(generated.split())
    ref_words = set(reference.split())
    if ref_words:
        scores["word_overlap"] = len(gen_words & ref_words) / len(gen_words | ref_words)
    else:
        scores["word_overlap"] = 0.0

    # Overall score (weighted average)
    scores["overall"] = (
        0.3 * scores["length_score"]
        + 0.4 * scores["number_overlap"]
        + 0.3 * scores["word_overlap"]
    )

    return scores


# ──────────────────────────────────────────────
# 4. Error Analysis
# ──────────────────────────────────────────────

def analyze_errors(results: list[dict]) -> dict:
    """
    Categorize failure modes from answer accuracy results.

    Categories:
      - repetition:   model repeats the same phrase
      - too_short:    generated answer is very short
      - wrong_number: model gave a number but it was wrong
      - no_number:    model didn't produce any number
      - correct:      answer matches
    """
    categories = Counter()
    examples = {
        "repetition": [],
        "too_short": [],
        "wrong_number": [],
        "no_number": [],
        "correct": [],
    }

    for r in results:
        gen = r["generated"]

        if r["correct"]:
            categories["correct"] += 1
            if len(examples["correct"]) < 3:
                examples["correct"].append(r)
            continue

        # Check repetition
        words = gen.split()
        if len(words) > 5:
            repeat_ratio = 1 - len(set(words)) / len(words)
            if repeat_ratio > 0.5:
                categories["repetition"] += 1
                if len(examples["repetition"]) < 3:
                    examples["repetition"].append(r)
                continue

        # Check too short
        if len(gen) < 20:
            categories["too_short"] += 1
            if len(examples["too_short"]) < 3:
                examples["too_short"].append(r)
            continue

        # Check number presence
        if r["pred_number"] is None:
            categories["no_number"] += 1
            if len(examples["no_number"]) < 3:
                examples["no_number"].append(r)
        else:
            categories["wrong_number"] += 1
            if len(examples["wrong_number"]) < 3:
                examples["wrong_number"].append(r)

    return {
        "category_counts": dict(categories),
        "total": len(results),
        "examples": examples,
    }


# ──────────────────────────────────────────────
# 5. Generation Quality Report
# ──────────────────────────────────────────────

def generation_quality_report(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 100,
) -> dict:
    """Generate text for prompts and compute quality statistics."""
    model.eval()
    generations = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        inp = torch.tensor([ids], device=device)
        gen_ids = generate(
            model, inp,
            max_new_tokens=max_new_tokens,
            context_length=model.cfg["context_length"],
            temperature=0.7,
            top_k=25,
        )
        gen_text = tokenizer.decode(gen_ids[0].tolist())
        generations.append(gen_text)

    # Compute stats
    lengths = [len(g.split()) for g in generations]
    unique_ratios = []
    for g in generations:
        words = g.split()
        unique_ratios.append(len(set(words)) / max(len(words), 1))

    return {
        "generations": [
            {"prompt": p, "generated": g}
            for p, g in zip(prompts, generations)
        ],
        "stats": {
            "avg_length_words": sum(lengths) / max(len(lengths), 1),
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "avg_unique_word_ratio": sum(unique_ratios) / max(len(unique_ratios), 1),
        },
    }


# ──────────────────────────────────────────────
# Full evaluation pipeline
# ──────────────────────────────────────────────

def run_full_evaluation(
    model: GPTModel,
    tokenizer: BPETokenizer,
    pretrain_loader,
    sft_data_path: str,
    device: torch.device,
    num_eval_samples: int = 50,
    save_dir: str | None = None,
) -> dict:
    """Run all evaluation metrics and save results."""
    save_dir = Path(save_dir or PROJECT_ROOT / "results")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Running Full Evaluation")
    print("="*60)

    # 1. Perplexity
    print("\n[1/4] Computing perplexity...")
    ppl = compute_perplexity(model, pretrain_loader, device)
    print(f"  Perplexity: {ppl:.2f}")

    # 2. Answer accuracy
    print("\n[2/4] Computing answer accuracy...")
    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    eval_samples = sft_data[:num_eval_samples]
    questions = [s["input"] for s in eval_samples]
    answers = [s["output"] for s in eval_samples]

    acc_results = compute_answer_accuracy(
        model, tokenizer, questions, answers, device
    )
    print(f"  Accuracy: {acc_results['accuracy']:.2%} "
          f"({acc_results['correct']}/{acc_results['total']})")

    # 3. Error analysis
    print("\n[3/4] Analyzing errors...")
    error_report = analyze_errors(acc_results["results"])
    print(f"  Error categories: {error_report['category_counts']}")

    # 4. Generation quality
    print("\n[4/4] Evaluating generation quality...")
    test_prompts = [
        "حل المسألة الرياضية التالية: إذا كان لدى سارة ١٢ كتاباً وأعطت ٤ لصديقتها",
        "ما هو حاصل جمع ٢٥ و ١٧؟",
        "اشترى محمد ٣ أقلام بسعر ٥ ريالات للقلم الواحد. كم دفع إجمالاً؟",
    ]
    quality_report = generation_quality_report(
        model, tokenizer, test_prompts, device
    )
    print(f"  Avg generation length: {quality_report['stats']['avg_length_words']:.0f} words")
    print(f"  Avg unique word ratio: {quality_report['stats']['avg_unique_word_ratio']:.2%}")

    # Save results
    full_results = {
        "perplexity": ppl,
        "answer_accuracy": {
            k: v for k, v in acc_results.items() if k != "results"
        },
        "error_analysis": error_report,
        "generation_quality": quality_report,
    }

    with open(save_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to {save_dir / 'evaluation_results.json'}")
    return full_results
