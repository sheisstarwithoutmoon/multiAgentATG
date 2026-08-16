"""
src/data/evaluator.py

Per-benchmark evaluation logic for the AHC framework.

DESIGN PRINCIPLE: No universal string-matching evaluator.
─────────────────────────────────────────────────────────
Each benchmark has distinct correctness semantics:

  GSM8K        : Extract final integer from model output; compare to GT integer.
                 String matching would fail on formatting ("60" vs "60.0").

  HaluEval QA  : GT is a factual phrase. Prediction must CONTAIN the GT
                 (case-insensitive substring match) — not exact equality.
                 The hallucinated_answer is a distractor, not the target.

  HaluEval Gen : GT is a binary label ("yes"/"no"). Model must output exactly
                 one of these tokens (after normalization). Substring matching
                 would incorrectly accept "yes, it is hallucinated".

  SimpleQA     : Normalized exact match. Case/punctuation/article stripped.
                 A universal substring check would give false positives
                 (e.g., "Paris" inside "Not Paris").

  FRAMES       : Factual multi-hop answer. Contains-match on key phrase,
                 but the threshold is stricter than HaluEval (full answer
                 phrase must appear, not just a keyword).

  MedHallu     : Same as HaluEval QA — factual answer phrase in prediction.

  MMLU-Pro     : Single letter A-J. Extract first letter from model output.
                 Never substring or contains — "A" in "BART" would match.

USAGE:
    from src.data.evaluator import evaluate_prediction

    result = evaluate_prediction(sample, prediction="The answer is 60")
    # result.correct: bool
    # result.score: float (0.0 or 1.0 for exact; may be fractional in future)
    # result.details: dict with per-benchmark diagnostic info
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.data.benchmark_schema import (
    BenchmarkSample,
    ANSWER_TYPE_NUMERIC,
    ANSWER_TYPE_EXACT_MATCH,
    ANSWER_TYPE_CONTAINS,
    ANSWER_TYPE_BINARY_LABEL,
    ANSWER_TYPE_LETTER_CHOICE,
    DATASET_GSM8K,
    DATASET_HALUEVAL,
    DATASET_HALUEVAL_GENERAL,
    DATASET_SIMPLEQA,
    DATASET_FRAMES,
    DATASET_MEDHALLU,
    DATASET_MMLU_PRO,
)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Structured result from a single-sample evaluation.

    Fields
    ------
    correct         : True if prediction matches ground truth.
    score           : 0.0 or 1.0 (binary correctness for all current benchmarks).
    ground_truth    : The GT value used for comparison (from sample, never from prompt).
    prediction      : The raw model prediction string.
    normalized_pred : The normalized/extracted prediction used for comparison.
    dataset         : Dataset name for logging.
    sample_id       : Sample ID for tracing.
    answer_type     : The evaluation method used.
    details         : Per-benchmark diagnostic information.
    error           : Non-None if evaluation failed (e.g., no number extracted).
    """
    correct: bool
    score: float
    ground_truth: str
    prediction: str
    normalized_pred: str
    dataset: str
    sample_id: str
    answer_type: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ─── Per-type normalization helpers ──────────────────────────────────────────

def _extract_numeric(text: str) -> Optional[float]:
    """
    Extracts the most likely final numeric answer from a model response.
    Mirrors _find_final_answer_number in injection.py but returns a float.
    """
    # 1. GSM8K canonical delimiter
    m = re.search(r"####\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))

    # 2. Explicit answer statement
    m = re.search(
        r"(?:answer|result|total|solution)\s+(?:is|=|:)\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)",
        text, re.IGNORECASE
    )
    if m:
        return float(m.group(1).replace(",", ""))

    # 3. Last standalone number in text (fallback)
    matches = list(re.finditer(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b", text))
    if matches:
        return float(matches[-1].group(1).replace(",", ""))

    return None


def _normalize_string(s: str) -> str:
    """
    Normalizes a string for exact-match comparison:
    - lowercase
    - strip leading/trailing whitespace
    - remove articles (a, an, the)
    - collapse multiple spaces
    - remove trailing punctuation
    """
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation at word boundaries
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_letter_choice(text: str) -> Optional[str]:
    """
    Extracts a single letter answer (A-J) from model output.

    Priority:
    1. Explicit patterns: "Answer: B", "The answer is C", "(D)"
    2. A lone letter on its own line
    3. The very first letter A-J in the text (last resort)
    """
    # Pattern 1: explicit answer labeling
    m = re.search(
        r"(?:answer|choice|option|select(?:ed)?)\s*(?:is|:)?\s*\(?([A-Ja-j])\)?",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Pattern 2: standalone letter on a line or in parentheses
    m = re.search(r"^\s*\(?([A-Ja-j])\)?\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).upper()

    # Pattern 3: first letter A-J in the text (last resort; risky)
    m = re.search(r"\b([A-Ja-j])\b", text)
    if m:
        return m.group(1).upper()

    return None


def _normalize_binary(text: str) -> Optional[str]:
    """
    Normalizes a model response to 'yes' or 'no'.
    Returns None if neither is clearly extractable.
    """
    t = text.strip().lower()
    if t in ("yes", "no"):
        return t
    # Look for yes/no as first word
    first_word = t.split()[0].strip(string.punctuation) if t.split() else ""
    if first_word in ("yes", "no"):
        return first_word
    return None


# ─── Per-benchmark evaluators ─────────────────────────────────────────────────

def _eval_numeric(sample: BenchmarkSample, prediction: str) -> EvaluationResult:
    """GSM8K and other numeric benchmarks."""
    extracted = _extract_numeric(prediction)
    if extracted is None:
        return EvaluationResult(
            correct=False, score=0.0,
            ground_truth=sample.ground_truth,
            prediction=prediction, normalized_pred="",
            dataset=sample.dataset, sample_id=sample.sample_id,
            answer_type=ANSWER_TYPE_NUMERIC,
            error="Could not extract numeric answer from prediction",
        )
    try:
        gt_val = float(sample.ground_truth.replace(",", ""))
    except ValueError:
        return EvaluationResult(
            correct=False, score=0.0,
            ground_truth=sample.ground_truth,
            prediction=prediction, normalized_pred=str(extracted),
            dataset=sample.dataset, sample_id=sample.sample_id,
            answer_type=ANSWER_TYPE_NUMERIC,
            error=f"Could not parse ground_truth as float: {sample.ground_truth!r}",
        )
    correct = abs(extracted - gt_val) < 1e-6  # exact for integers; tolerance for floats
    return EvaluationResult(
        correct=correct, score=1.0 if correct else 0.0,
        ground_truth=sample.ground_truth,
        prediction=prediction, normalized_pred=str(extracted),
        dataset=sample.dataset, sample_id=sample.sample_id,
        answer_type=ANSWER_TYPE_NUMERIC,
        details={"extracted": extracted, "gt_float": gt_val},
    )


def _eval_exact_match(sample: BenchmarkSample, prediction: str) -> EvaluationResult:
    """SimpleQA and other exact-match benchmarks."""
    norm_pred = _normalize_string(prediction)
    norm_gt = _normalize_string(sample.ground_truth)
    correct = norm_pred == norm_gt
    return EvaluationResult(
        correct=correct, score=1.0 if correct else 0.0,
        ground_truth=sample.ground_truth,
        prediction=prediction, normalized_pred=norm_pred,
        dataset=sample.dataset, sample_id=sample.sample_id,
        answer_type=ANSWER_TYPE_EXACT_MATCH,
        details={"normalized_gt": norm_gt},
    )


def _eval_contains(sample: BenchmarkSample, prediction: str) -> EvaluationResult:
    """HaluEval QA, FRAMES, MedHallu — ground truth phrase must appear in prediction."""
    norm_pred = prediction.lower().strip()
    norm_gt = sample.ground_truth.lower().strip()
    correct = norm_gt in norm_pred
    return EvaluationResult(
        correct=correct, score=1.0 if correct else 0.0,
        ground_truth=sample.ground_truth,
        prediction=prediction, normalized_pred=norm_pred,
        dataset=sample.dataset, sample_id=sample.sample_id,
        answer_type=ANSWER_TYPE_CONTAINS,
        details={"gt_lower": norm_gt, "found_at": norm_pred.find(norm_gt)},
    )


def _eval_binary_label(sample: BenchmarkSample, prediction: str) -> EvaluationResult:
    """HaluEval General — binary yes/no classification."""
    norm_pred = _normalize_binary(prediction)
    norm_gt = sample.ground_truth.strip().lower()
    if norm_pred is None:
        return EvaluationResult(
            correct=False, score=0.0,
            ground_truth=sample.ground_truth,
            prediction=prediction, normalized_pred="",
            dataset=sample.dataset, sample_id=sample.sample_id,
            answer_type=ANSWER_TYPE_BINARY_LABEL,
            error="Could not extract yes/no from prediction",
        )
    correct = norm_pred == norm_gt
    return EvaluationResult(
        correct=correct, score=1.0 if correct else 0.0,
        ground_truth=sample.ground_truth,
        prediction=prediction, normalized_pred=norm_pred,
        dataset=sample.dataset, sample_id=sample.sample_id,
        answer_type=ANSWER_TYPE_BINARY_LABEL,
    )


def _eval_letter_choice(sample: BenchmarkSample, prediction: str) -> EvaluationResult:
    """MMLU-Pro — single letter A-J."""
    extracted = _extract_letter_choice(prediction)
    if extracted is None:
        return EvaluationResult(
            correct=False, score=0.0,
            ground_truth=sample.ground_truth,
            prediction=prediction, normalized_pred="",
            dataset=sample.dataset, sample_id=sample.sample_id,
            answer_type=ANSWER_TYPE_LETTER_CHOICE,
            error="Could not extract letter choice from prediction",
        )
    correct = extracted == sample.ground_truth.upper()
    return EvaluationResult(
        correct=correct, score=1.0 if correct else 0.0,
        ground_truth=sample.ground_truth,
        prediction=prediction, normalized_pred=extracted,
        dataset=sample.dataset, sample_id=sample.sample_id,
        answer_type=ANSWER_TYPE_LETTER_CHOICE,
        details={
            "correct_option_text": (
                sample.options[sample.answer_index]
                if sample.options and sample.answer_index is not None
                else None
            ),
        },
    )


# ─── Dispatch table ───────────────────────────────────────────────────────────

# Maps answer_type → evaluator function.
# This is the ONLY place where the dispatch happens.
# Adding a new benchmark requires: (1) add a constant, (2) add an entry here.
_EVALUATOR_DISPATCH = {
    ANSWER_TYPE_NUMERIC:      _eval_numeric,
    ANSWER_TYPE_EXACT_MATCH:  _eval_exact_match,
    ANSWER_TYPE_CONTAINS:     _eval_contains,
    ANSWER_TYPE_BINARY_LABEL: _eval_binary_label,
    ANSWER_TYPE_LETTER_CHOICE: _eval_letter_choice,
}


def evaluate_prediction(
    sample: BenchmarkSample,
    prediction: str,
) -> EvaluationResult:
    """
    Evaluates a model prediction against a BenchmarkSample using the
    benchmark-specific evaluator for the sample's answer_type.

    This is the ONLY function downstream code should call for evaluation.
    It dispatches to the correct per-type evaluator automatically.

    Args:
        sample     : BenchmarkSample with ground_truth and answer_type set.
        prediction : Raw string output from the model.

    Returns:
        EvaluationResult with correctness, score, normalized values, and details.

    Raises:
        ValueError : If sample.answer_type is not in EVAL_ONLY_FIELDS.
    """
    evaluator = _EVALUATOR_DISPATCH.get(sample.answer_type)
    if evaluator is None:
        raise ValueError(
            f"No evaluator registered for answer_type={sample.answer_type!r}. "
            f"Registered types: {sorted(_EVALUATOR_DISPATCH.keys())}"
        )
    return evaluator(sample, prediction)


def get_supported_answer_types() -> list:
    """Returns the list of answer types with registered evaluators."""
    return sorted(_EVALUATOR_DISPATCH.keys())
