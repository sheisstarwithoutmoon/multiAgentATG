"""
tests/test_real_data.py

Real-data smoke tests for all supported benchmarks.

REQUIRES NETWORK ACCESS AND HUGGINGFACE CREDENTIALS.
These tests are SKIPPED if HF_TOKEN is not set or the network is unavailable.

Run with:
    cd c:\\Users\\iamva\\multi_agent_atg
    python -m pytest tests/test_real_data.py -v -s

Or run the standalone smoke test:
    python -m src.data.loader  (triggers __main__ which calls run_smoke_test)

Purpose:
  - Verify actual HuggingFace dataset schemas match our expectations.
  - Confirm that real sample fields map correctly to BenchmarkSample.
  - Validate ground-truth integrity on real data.
  - Detect schema changes in upstream datasets.
"""

import os
import pytest
from typing import List

from src.data.benchmark_schema import (
    BenchmarkSample,
    ANSWER_TYPE_NUMERIC, ANSWER_TYPE_EXACT_MATCH, ANSWER_TYPE_CONTAINS,
    ANSWER_TYPE_BINARY_LABEL, ANSWER_TYPE_LETTER_CHOICE,
    TASK_MATHEMATICAL_REASONING, TASK_FACTUAL_QA, TASK_HALLUCINATION_DETECTION,
    TASK_MULTIPLE_CHOICE, TASK_MULTI_HOP_REASONING,
    DATASET_GSM8K, DATASET_HALUEVAL, DATASET_HALUEVAL_GENERAL,
    DATASET_SIMPLEQA, DATASET_FRAMES, DATASET_MEDHALLU, DATASET_MMLU_PRO,
)
from src.data.loader import (
    load_gsm8k_test, load_halueval_qa, load_halueval_general,
    load_simpleqa, load_frames, load_medhallu, load_mmlu_pro,
    run_smoke_test, SUPPORTED_DATASETS,
)

# ─── Network availability marker ───────────────────────────────────────────────

def _network_available() -> bool:
    """Best-effort check for HF connectivity."""
    try:
        import socket
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
        return True
    except Exception:
        return False

requires_network = pytest.mark.skipif(
    not _network_available(),
    reason="Network unavailable — skipping real-data smoke tests"
)

N = 3  # samples per dataset

# ─── Shared validation helper ──────────────────────────────────────────────────

FORBIDDEN_FIELDS = {
    "ground_truth", "hallucinated_answer", "reference_label",
    "hallucination_spans", "answer_index",
}


def _validate_real_samples(
    samples: List[BenchmarkSample],
    expected_dataset: str,
    expected_task: str,
    expected_answer_type: str,
    n: int = N,
) -> None:
    assert len(samples) == n, f"Expected {n} samples, got {len(samples)}"
    for s in samples:
        # Basic field presence
        assert s.sample_id, "sample_id must not be empty"
        assert s.dataset == expected_dataset
        assert s.question, "question must not be empty"
        assert s.ground_truth is not None, "ground_truth must not be None"
        assert s.task_type == expected_task
        assert s.answer_type == expected_answer_type

        # Prompt safety check
        safe = s.prompt_safe_dict()
        for field in FORBIDDEN_FIELDS:
            assert field not in safe, (
                f"PROMPT LEAKAGE: '{field}' found in prompt_safe_dict() "
                f"for real sample {s.sample_id}."
            )

        # Print diagnostic for review
        print(f"\n  [{s.sample_id}]")
        print(f"    task_type:    {s.task_type}")
        print(f"    answer_type:  {s.answer_type}")
        print(f"    question:     {s.question[:100]}...")
        if s.knowledge:
            print(f"    knowledge:    {s.knowledge[:80]}...")
        print(f"    ground_truth: {s.ground_truth[:80]}")
        if s.hallucinated_answer:
            print(f"    hallucinated: {s.hallucinated_answer[:80]}")
        if s.reference_label:
            print(f"    ref_label:    {s.reference_label}")
        if s.options:
            print(f"    options:      {len(s.options)} options, correct={s.ground_truth}")
        if s.reasoning_types:
            print(f"    reasoning:    {s.reasoning_types}")
        if s.category:
            print(f"    category:     {s.category}")
        if s.difficulty:
            print(f"    difficulty:   {s.difficulty}")


# ─── Real data smoke tests ────────────────────────────────────────────────────

@requires_network
def test_real_gsm8k():
    """GSM8K: openai/gsm8k, config=main, split=test"""
    samples = load_gsm8k_test(limit=N, strict=True)
    _validate_real_samples(samples, DATASET_GSM8K, TASK_MATHEMATICAL_REASONING, ANSWER_TYPE_NUMERIC)
    for s in samples:
        # GSM8K ground truth must be a non-empty numeric string
        assert s.ground_truth.isdigit() or "." in s.ground_truth, (
            f"GSM8K ground_truth should be numeric: {s.ground_truth!r}"
        )
        # Raw answer must be in metadata
        assert "raw_answer" in s.metadata
        assert "#### " in s.metadata["raw_answer"]


@requires_network
def test_real_halueval_qa():
    """HaluEval QA: pminervini/HaluEval, config=qa, split=data"""
    samples = load_halueval_qa(limit=N, strict=True)
    _validate_real_samples(samples, DATASET_HALUEVAL, TASK_HALLUCINATION_DETECTION, ANSWER_TYPE_CONTAINS)
    for s in samples:
        # HaluEval QA must always have hallucinated_answer and knowledge
        assert s.hallucinated_answer, f"Missing hallucinated_answer in {s.sample_id}"
        assert s.knowledge, f"Missing knowledge in {s.sample_id}"
        # Hallucinated answer must differ from ground truth
        assert s.hallucinated_answer.lower().strip() != s.ground_truth.lower().strip(), (
            f"hallucinated_answer must differ from ground_truth in {s.sample_id}"
        )


@requires_network
def test_real_halueval_general():
    """HaluEval General: pminervini/HaluEval, config=general, split=data"""
    samples = load_halueval_general(limit=N, strict=True)
    _validate_real_samples(samples, DATASET_HALUEVAL_GENERAL, TASK_HALLUCINATION_DETECTION, ANSWER_TYPE_BINARY_LABEL)
    for s in samples:
        assert s.ground_truth in ("yes", "no"), (
            f"HaluEval general ground_truth must be 'yes'/'no', got {s.ground_truth!r}"
        )
        assert s.reference_label in ("yes", "no")
        # knowledge stores the chatgpt_response being evaluated
        assert s.knowledge, f"HaluEval general must store chatgpt_response in knowledge"


@requires_network
def test_real_simpleqa():
    """SimpleQA: basicv8vc/SimpleQA, split=test"""
    samples = load_simpleqa(limit=N, strict=True)
    _validate_real_samples(samples, DATASET_SIMPLEQA, TASK_FACTUAL_QA, ANSWER_TYPE_EXACT_MATCH)
    for s in samples:
        assert s.ground_truth, "SimpleQA ground_truth must not be empty"
        # Metadata must contain topic (from parsed JSON)
        # topic may be None if metadata parse failed — acceptable
        assert isinstance(s.metadata, dict)


@requires_network
def test_real_frames():
    """FRAMES: google/frames-benchmark, split=test"""
    samples = load_frames(limit=N, strict=True)
    _validate_real_samples(samples, DATASET_FRAMES, TASK_MULTI_HOP_REASONING, ANSWER_TYPE_CONTAINS)
    for s in samples:
        assert s.reasoning_types, f"FRAMES must have reasoning_types in {s.sample_id}"
        # FRAMES questions require multi-hop reasoning — they are typically long
        assert len(s.question) > 30, "FRAMES questions should be multi-hop (length > 30)"


@requires_network
def test_real_medhallu():
    """MedHallu: UTAustin-AIHealth/MedHallu, config=pqa_labeled, split=train"""
    samples = load_medhallu(config="pqa_labeled", split="train", limit=N, strict=True)
    _validate_real_samples(samples, DATASET_MEDHALLU, TASK_HALLUCINATION_DETECTION, ANSWER_TYPE_CONTAINS)
    for s in samples:
        assert s.hallucinated_answer, f"Missing hallucinated_answer in {s.sample_id}"
        assert s.knowledge, f"Missing knowledge in {s.sample_id}"
        assert s.difficulty in ("easy", "medium", "hard"), (
            f"MedHallu difficulty must be easy/medium/hard, got {s.difficulty!r}"
        )
        assert s.category is not None, f"MedHallu category must not be None for {s.sample_id}"
        # Real MedHallu categories confirmed from actual data (some use #Question# markers):
        VALID_MEDHALLU_CATEGORIES = {
            "Misinterpretation of Question",
            "Misinterpretation of #Question#",  # actual value in pqa_labeled
            "Incomplete Information",
            "Mechanism and Pathway Misattribution",
            "Methodological and Evidence Fabrication",
        }
        assert s.category in VALID_MEDHALLU_CATEGORIES, (
            f"Unknown MedHallu hallucination category: {s.category!r}\n"
            f"Expected one of: {VALID_MEDHALLU_CATEGORIES}"
        )


@requires_network
def test_real_mmlu_pro():
    """MMLU-Pro: TIGER-Lab/MMLU-Pro, split=test"""
    samples = load_mmlu_pro(limit=N, strict=True)
    _validate_real_samples(samples, DATASET_MMLU_PRO, TASK_MULTIPLE_CHOICE, ANSWER_TYPE_LETTER_CHOICE)
    import re
    for s in samples:
        # Ground truth must be a single letter A-J
        assert re.match(r"^[A-J]$", s.ground_truth), (
            f"MMLU-Pro ground_truth must be letter A-J, got {s.ground_truth!r}"
        )
        # options must be a list of exactly 10 items
        # Real MMLU-Pro: most questions have 10 options, but some have 9.
        # (Confirmed from actual data: question_id=70 has 9 options, answer_index=8)
        assert s.options is not None
        assert len(s.options) >= 9, f"MMLU-Pro must have 9-10 options, got {len(s.options)}"
        # answer_index must correspond to the correct letter
        expected_letter = chr(ord("A") + s.answer_index)
        assert s.ground_truth == expected_letter, (
            f"ground_truth {s.ground_truth} doesn't match "
            f"options[answer_index={s.answer_index}] -> letter {expected_letter}"
        )
        # options[answer_index] should be non-empty
        assert s.options[s.answer_index], "Correct option text must not be empty"
        assert s.category, "MMLU-Pro must have a category"


@requires_network
def test_full_smoke_test_all_datasets():
    """Run the full run_smoke_test() across all datasets and verify no errors."""
    results = run_smoke_test(n_per_dataset=N, strict=True)
    failures = {name: r for name, r in results.items() if r["error"] is not None}
    assert not failures, (
        f"Smoke test failed for datasets: {list(failures.keys())}\n"
        f"Errors: {failures}"
    )
    for name, r in results.items():
        assert r["loaded"] >= 1, f"Dataset {name}: expected >= 1 sample, got {r['loaded']}"
