"""
Detailed Error Analysis for Arabic Math Reasoning.

Provides deeper analysis of model failures including:
  - Repetition detection
  - Context loss analysis
  - Mathematical reasoning failure classification
  - Visualization helpers for error reports
"""

import re
from collections import Counter
from src.evaluation.metrics import ARABIC_DIGITS, extract_number


def detect_repetition(text: str, n: int = 3) -> dict:
    """Detect n-gram repetition in generated text."""
    words = text.split()
    if len(words) < n:
        return {"is_repetitive": False, "repetition_ratio": 0.0, "repeated_phrases": []}

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)

    repeated = {
        " ".join(ng): count
        for ng, count in ngram_counts.items()
        if count > 2
    }

    total_ngrams = len(ngrams)
    repeated_ngrams = sum(c - 1 for c in ngram_counts.values() if c > 1)
    ratio = repeated_ngrams / max(total_ngrams, 1)

    return {
        "is_repetitive": ratio > 0.3,
        "repetition_ratio": ratio,
        "repeated_phrases": list(repeated.keys())[:5],
    }


def classify_math_error(question: str, gold: str, generated: str) -> str:
    """
    Classify the type of mathematical reasoning error.

    Categories:
      - correct:           answer matches
      - arithmetic_error:  correct approach but wrong calculation
      - missing_steps:     skipped intermediate reasoning
      - wrong_operation:   used wrong math operation
      - hallucination:     generated irrelevant content
      - no_answer:         didn't produce a final answer
      - repetition:        got stuck in a loop
    """
    gold_num = extract_number(gold)
    pred_num = extract_number(generated)

    # Correct
    if pred_num is not None and gold_num is not None:
        if abs(pred_num - gold_num) < 1e-6:
            return "correct"

    # Repetition
    rep = detect_repetition(generated)
    if rep["is_repetitive"]:
        return "repetition"

    # No answer
    if pred_num is None:
        if len(generated.strip()) < 10:
            return "no_answer"
        return "hallucination"

    # Has a number but wrong — try to classify further
    gen_normalized = generated.translate(ARABIC_DIGITS)

    ops_in_gold = {
        "+": "جمع" in gold or "+" in gold or "زائد" in gold,
        "-": "طرح" in gold or "-" in gold or "ناقص" in gold,
        "*": "ضرب" in gold or "×" in gold or "في" in gold,
        "/": "قسمة" in gold or "÷" in gold or "على" in gold,
    }
    ops_in_gen = {
        "+": "جمع" in generated or "+" in generated or "زائد" in generated,
        "-": "طرح" in generated or "-" in generated or "ناقص" in generated,
        "*": "ضرب" in generated or "×" in generated or "في" in generated,
        "/": "قسمة" in generated or "÷" in generated or "على" in generated,
    }

    gold_ops = {k for k, v in ops_in_gold.items() if v}
    gen_ops = {k for k, v in ops_in_gen.items() if v}

    if gold_ops and gen_ops and gold_ops != gen_ops:
        return "wrong_operation"

    # Check step count
    gold_steps = len(re.findall(r"\d+", gold.translate(ARABIC_DIGITS)))
    gen_steps = len(re.findall(r"\d+", gen_normalized))

    if gen_steps < gold_steps * 0.5:
        return "missing_steps"

    return "arithmetic_error"


def generate_error_report(results: list[dict]) -> dict:
    """
    Generate a comprehensive error report with classification and examples.
    """
    classifications = Counter()
    classified_results = []

    for r in results:
        error_type = classify_math_error(
            r["question"], r["gold_answer"], r["generated"]
        )
        classifications[error_type] += 1
        classified_results.append({**r, "error_type": error_type})

    examples_by_type = {}
    for error_type in classifications:
        examples_by_type[error_type] = [
            r for r in classified_results if r["error_type"] == error_type
        ][:3]

    return {
        "classification_counts": dict(classifications),
        "total": len(results),
        "accuracy": classifications.get("correct", 0) / max(len(results), 1),
        "examples_by_type": examples_by_type,
        "detailed_results": classified_results,
    }
