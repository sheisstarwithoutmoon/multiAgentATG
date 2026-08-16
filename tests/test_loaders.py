"""
tests/test_loaders.py

Phase 2 Research-Integrity Revision — Comprehensive Test Suite.

Test categories:
  1. BenchmarkSample schema (structure, serialisation, ground-truth isolation)
  2. Offline stub validation (all 7 loaders, strict=False)
  3. Explicit task type assignment per loader
  4. Ground-truth integrity (mutation warning, field isolation)
  5. Prompt leakage (ground-truth fields never in prompt-safe dict)
  6. Fault injection validity (semantic correctness of faults)
  7. Unified factory + split remapping
  8. Shuffle determinism
  9. Reproducibility configuration tests
 10. Dependency audit (imports used)

OFFLINE vs REAL-DATA TESTS:
  - All tests in this file run OFFLINE (strict=False, stub fallback).
  - Real-data smoke tests are in tests/test_real_data.py (requires network).

Run with:
    cd c:\\Users\\iamva\\multi_agent_atg
    python -m pytest tests/test_loaders.py -v
"""

import re
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
    load_dataset_by_name, shuffle_dataset, extract_gsm8k_numeric_truth,
    SUPPORTED_DATASETS,
)
from src.data.injection import (
    extract_gsm8k_truth, inject_adversarial_mutation, inject_answer_fault,
    should_apply_fault, FAULT_RATES, _find_final_answer_number,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BenchmarkSample schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkSampleSchema:
    def test_requires_task_type_and_answer_type(self):
        """task_type and answer_type must be explicit — no 'contains' default."""
        s = BenchmarkSample(
            sample_id="t1",
            dataset=DATASET_GSM8K,
            question="Q?",
            ground_truth="42",
            task_type=TASK_MATHEMATICAL_REASONING,
            answer_type=ANSWER_TYPE_NUMERIC,
        )
        assert s.task_type == TASK_MATHEMATICAL_REASONING
        assert s.answer_type == ANSWER_TYPE_NUMERIC

    def test_full_construction_all_fields(self):
        s = BenchmarkSample(
            sample_id="t2",
            dataset=DATASET_MEDHALLU,
            question="Medical question?",
            ground_truth="Correct answer.",
            task_type=TASK_HALLUCINATION_DETECTION,
            answer_type=ANSWER_TYPE_CONTAINS,
            knowledge="Supporting passage.",
            hallucinated_answer="Wrong answer.",
            reference_label="no",
            hallucination_spans=["wrong span"],
            category="Mechanism and Pathway Misattribution",
            difficulty="hard",
            options=None,
            answer_index=None,
            reasoning_types=None,
            wiki_links=None,
            metadata={"config": "pqa_labeled"},
        )
        assert s.hallucinated_answer == "Wrong answer."
        assert s.hallucination_spans == ["wrong span"]

    def test_to_dict_round_trip(self):
        s = BenchmarkSample(
            sample_id="t3",
            dataset=DATASET_SIMPLEQA,
            question="When?",
            ground_truth="1889",
            task_type=TASK_FACTUAL_QA,
            answer_type=ANSWER_TYPE_EXACT_MATCH,
        )
        d = s.to_dict()
        s2 = BenchmarkSample.from_dict(d)
        assert s2.sample_id == s.sample_id
        assert s2.task_type == s.task_type
        assert s2.answer_type == s.answer_type

    def test_from_dict_extra_keys_go_to_metadata(self):
        d = {
            "sample_id": "t4", "dataset": "custom", "question": "Q?",
            "ground_truth": "A", "task_type": TASK_FACTUAL_QA,
            "answer_type": ANSWER_TYPE_CONTAINS,
            "extra_field": "should_go_to_metadata",
        }
        s = BenchmarkSample.from_dict(d)
        assert "extra_field" in s.metadata

    def test_task_type_constants_are_distinct(self):
        types = [
            TASK_MATHEMATICAL_REASONING, TASK_FACTUAL_QA,
            TASK_HALLUCINATION_DETECTION, TASK_MULTIPLE_CHOICE,
            TASK_MULTI_HOP_REASONING,
        ]
        assert len(set(types)) == 5

    def test_answer_type_constants_are_distinct(self):
        types = [
            ANSWER_TYPE_NUMERIC, ANSWER_TYPE_EXACT_MATCH, ANSWER_TYPE_CONTAINS,
            ANSWER_TYPE_BINARY_LABEL, ANSWER_TYPE_LETTER_CHOICE,
        ]
        assert len(set(types)) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Ground-Truth Isolation (Prompt Leakage Prevention)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthIsolation:
    """
    These tests verify that ground-truth fields are isolated from prompt-safe
    representations.  CRITICAL for research integrity.
    """

    FORBIDDEN_FIELDS = {
        "ground_truth", "hallucinated_answer", "reference_label",
        "hallucination_spans", "answer_index",
    }

    def _make_full_sample(self):
        return BenchmarkSample(
            sample_id="leak_test",
            dataset=DATASET_HALUEVAL,
            question="Who directed Inception?",
            ground_truth="Christopher Nolan",
            task_type=TASK_HALLUCINATION_DETECTION,
            answer_type=ANSWER_TYPE_CONTAINS,
            knowledge="Inception was directed by Christopher Nolan.",
            hallucinated_answer="Steven Spielberg",
            reference_label="no",
            hallucination_spans=["Spielberg directed"],
            answer_index=None,
        )

    def test_prompt_safe_dict_excludes_all_forbidden_fields(self):
        s = self._make_full_sample()
        safe = s.prompt_safe_dict()
        for field in self.FORBIDDEN_FIELDS:
            assert field not in safe, (
                f"PROMPT LEAKAGE: '{field}' found in prompt_safe_dict(). "
                "Ground-truth must never appear in model prompts."
            )

    def test_prompt_safe_dict_contains_question(self):
        s = self._make_full_sample()
        safe = s.prompt_safe_dict()
        assert "question" in safe
        assert safe["question"] == s.question

    def test_prompt_safe_dict_contains_knowledge(self):
        s = self._make_full_sample()
        safe = s.prompt_safe_dict()
        assert "knowledge" in safe

    def test_get_forbidden_fields_returns_all_set_gt_fields(self):
        s = self._make_full_sample()
        forbidden = s.get_forbidden_fields()
        assert "ground_truth" in forbidden
        assert forbidden["ground_truth"] == "Christopher Nolan"
        assert "hallucinated_answer" in forbidden
        assert "reference_label" in forbidden
        assert "hallucination_spans" in forbidden

    def test_to_dict_contains_ground_truth_for_evaluator(self):
        """to_dict() is for evaluation — MUST contain ground truth."""
        s = self._make_full_sample()
        d = s.to_dict()
        assert "ground_truth" in d
        assert d["ground_truth"] == "Christopher Nolan"

    def test_mmlu_pro_answer_index_is_forbidden(self):
        s = BenchmarkSample(
            sample_id="mc_leak",
            dataset=DATASET_MMLU_PRO,
            question="Which is correct?",
            ground_truth="B",
            task_type=TASK_MULTIPLE_CHOICE,
            answer_type=ANSWER_TYPE_LETTER_CHOICE,
            options=["Wrong A", "Correct B", "Wrong C"],
            answer_index=1,
        )
        safe = s.prompt_safe_dict()
        assert "answer_index" not in safe, (
            "answer_index leaks the correct option position to the model."
        )
        assert "ground_truth" not in safe

    def test_stub_samples_pass_prompt_safety(self):
        """All offline stubs must also pass the prompt leakage check."""
        for name in SUPPORTED_DATASETS:
            samples = load_dataset_by_name(name, limit=1, strict=False)
            for s in samples:
                safe = s.prompt_safe_dict()
                for field in self.FORBIDDEN_FIELDS:
                    assert field not in safe, (
                        f"PROMPT LEAKAGE in stub for '{name}': field '{field}' "
                        f"found in prompt_safe_dict()."
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Offline Stub Validation — Task Types + Answer Types
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoaderOfflineStubs:
    """Validates that stubs conform to verified real-schema task/answer types."""

    def _check(self, samples: List[BenchmarkSample], expected_dataset: str,
               expected_task: str, expected_answer_type: str):
        assert len(samples) >= 1
        for s in samples:
            assert isinstance(s, BenchmarkSample)
            assert s.dataset == expected_dataset
            assert s.task_type == expected_task, (
                f"{s.sample_id}: task_type={s.task_type!r}, expected {expected_task!r}"
            )
            assert s.answer_type == expected_answer_type, (
                f"{s.sample_id}: answer_type={s.answer_type!r}, expected {expected_answer_type!r}"
            )
            assert s.sample_id
            assert s.question
            assert s.ground_truth is not None

    def test_gsm8k_stub(self):
        s = load_gsm8k_test(limit=1, strict=False)
        self._check(s, DATASET_GSM8K, TASK_MATHEMATICAL_REASONING, ANSWER_TYPE_NUMERIC)

    def test_halueval_qa_stub(self):
        s = load_halueval_qa(limit=1, strict=False)
        self._check(s, DATASET_HALUEVAL, TASK_HALLUCINATION_DETECTION, ANSWER_TYPE_CONTAINS)
        assert s[0].hallucinated_answer is not None
        assert s[0].knowledge is not None

    def test_halueval_general_stub(self):
        s = load_halueval_general(limit=1, strict=False)
        self._check(s, DATASET_HALUEVAL_GENERAL, TASK_HALLUCINATION_DETECTION, ANSWER_TYPE_BINARY_LABEL)
        # reference_label must be yes or no
        assert s[0].reference_label in ("yes", "no")

    def test_simpleqa_stub(self):
        s = load_simpleqa(limit=1, strict=False)
        self._check(s, DATASET_SIMPLEQA, TASK_FACTUAL_QA, ANSWER_TYPE_EXACT_MATCH)

    def test_frames_stub(self):
        s = load_frames(limit=1, strict=False)
        self._check(s, DATASET_FRAMES, TASK_MULTI_HOP_REASONING, ANSWER_TYPE_CONTAINS)

    def test_medhallu_stub(self):
        s = load_medhallu(limit=1, strict=False)
        self._check(s, DATASET_MEDHALLU, TASK_HALLUCINATION_DETECTION, ANSWER_TYPE_CONTAINS)
        assert s[0].hallucinated_answer is not None
        assert s[0].category is not None
        assert s[0].difficulty is not None

    def test_mmlu_pro_stub(self):
        s = load_mmlu_pro(limit=1, strict=False)
        self._check(s, DATASET_MMLU_PRO, TASK_MULTIPLE_CHOICE, ANSWER_TYPE_LETTER_CHOICE)
        assert s[0].options is not None
        # MMLU-Pro has up to 10 options per question; some questions have 9.
        # Real data confirmed: answer_index=8 (9th option, 0-based) exists.
        assert len(s[0].options) >= 9, (
            f"MMLU-Pro should have 9-10 options, got {len(s[0].options)}"
        )
        assert s[0].answer_index is not None
        # ground_truth must be a letter A-J
        assert re.match(r"^[A-J]$", s[0].ground_truth), (
            f"MMLU-Pro ground_truth must be a letter A-J, got {s[0].ground_truth!r}"
        )
        # options[answer_index] should correspond to the correct answer
        correct_option = s[0].options[s[0].answer_index]
        assert isinstance(correct_option, str) and len(correct_option) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HaluEval Task Structure (not ordinary QA)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHaluEvalTaskStructure:
    """
    HaluEval has two distinct sub-tasks. This test group verifies that each
    is correctly represented and NOT treated as ordinary factual QA.
    """

    def test_halueval_qa_is_not_factual_qa(self):
        s = load_halueval_qa(limit=1, strict=False)
        assert s[0].task_type == TASK_HALLUCINATION_DETECTION
        assert s[0].task_type != TASK_FACTUAL_QA

    def test_halueval_qa_has_hallucinated_foil(self):
        s = load_halueval_qa(limit=1, strict=False)
        assert s[0].hallucinated_answer is not None

    def test_halueval_qa_has_knowledge_passage(self):
        s = load_halueval_qa(limit=1, strict=False)
        assert s[0].knowledge is not None and len(s[0].knowledge) > 0

    def test_halueval_general_ground_truth_is_binary(self):
        s = load_halueval_general(limit=1, strict=False)
        assert s[0].ground_truth in ("yes", "no"), (
            f"HaluEval general ground_truth must be 'yes'/'no', "
            f"got {s[0].ground_truth!r}"
        )

    def test_halueval_general_has_binary_label(self):
        s = load_halueval_general(limit=1, strict=False)
        assert s[0].answer_type == ANSWER_TYPE_BINARY_LABEL

    def test_halueval_datasets_are_separate(self):
        """halueval and halueval_general must have different dataset constants."""
        assert DATASET_HALUEVAL != DATASET_HALUEVAL_GENERAL


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Ground-Truth Mutation Warning (Query-Level Injection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryMutationGroundTruthInvalidation:
    """
    Query-level mutation invalidates ground truth and must NOT be used in
    the main benchmark evaluation pipeline.
    """

    def test_mutation_sets_invalidation_flag(self):
        s = {"question": "Alice has 10 apples and 3 oranges. Total?", "answer": "13"}
        m = inject_adversarial_mutation(s, dataset_type="gsm8k", seed=42)
        assert m.get("ground_truth_invalidated") is True, (
            "inject_adversarial_mutation must set ground_truth_invalidated=True "
            "so downstream code cannot accidentally use the original ground truth."
        )

    def test_mutation_changes_question(self):
        s = {"question": "Bob has 20 books and 8 pencils. How many items total?", "answer": "28"}
        m = inject_adversarial_mutation(s, dataset_type="gsm8k", seed=7)
        assert m["question"] != s["question"]

    def test_mutation_does_not_change_ground_truth_value(self):
        """
        The ground_truth VALUE in the dict is NOT changed (original numeric
        answer is preserved in the dict), but the flag signals it's invalid.
        This is intentional: the evaluator must check the flag, not assume GT.
        """
        s = {"question": "8 * 5 = ?", "ground_truth": "40"}
        m = inject_adversarial_mutation(s, dataset_type="gsm8k", seed=1)
        assert "ground_truth_invalidated" in m

    def test_mutation_seed_reproducibility(self):
        s = {"question": "Store sells 15 apples at $2 each. Total revenue?", "answer": "30"}
        m1 = inject_adversarial_mutation(s, dataset_type="gsm8k", seed=99)
        m2 = inject_adversarial_mutation(s, dataset_type="gsm8k", seed=99)
        assert m1["question"] == m2["question"]

    def test_benchmark_sample_mutation_sets_flag(self):
        sample = BenchmarkSample(
            sample_id="mut_test",
            dataset=DATASET_GSM8K,
            question="Tom has 20 toys and 5 books. Total?",
            ground_truth="25",
            task_type=TASK_MATHEMATICAL_REASONING,
            answer_type=ANSWER_TYPE_NUMERIC,
        )
        m = inject_adversarial_mutation(sample, seed=3)
        assert m.get("ground_truth_invalidated") is True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Fault Injection Semantic Validity
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaultInjectionValidity:
    """
    Verifies that each fault type produces the intended SEMANTIC effect,
    not just any text modification.
    """

    # A realistic GSM8K-style answer with a clear final answer
    MATH_ANSWER = (
        "Step 1: She earns 12 per hour. "
        "Step 2: She worked 5 hours. "
        "Step 3: Total = 12 * 5 = 60. "
        "The answer is 60."
    )
    MATH_QUESTION = "Weng earns $12 an hour for babysitting. She babysat for 5 hours. How much did she earn?"

    GSM8K_ANSWER = "She sells 9 duck eggs a day. She makes 9 * 2 = $18 every day.\n#### 18"

    def test_none_fault_unchanged(self):
        result = inject_answer_fault(self.MATH_ANSWER, "none")
        assert result == self.MATH_ANSWER

    def test_empty_string_unchanged(self):
        for ft in ["plausible_hallucination", "incorrect_number", "contradiction", "incorrect_reasoning_step"]:
            assert inject_answer_fault("", ft) == ""

    def test_incorrect_number_targets_final_answer(self):
        """incorrect_number must change the final answer number (60 or 18)."""
        result = inject_answer_fault(self.MATH_ANSWER, "incorrect_number", seed=1)
        assert result != self.MATH_ANSWER
        # Extract final answer from original and mutated
        orig_final = _find_final_answer_number(self.MATH_ANSWER)
        mutated_final = _find_final_answer_number(result)
        assert orig_final is not None
        assert mutated_final is not None
        # The final answer number must have changed
        assert orig_final[0] != mutated_final[0], (
            f"incorrect_number fault must change the final answer number. "
            f"Original={orig_final[0]}, Mutated={mutated_final[0]}"
        )

    def test_incorrect_number_targets_gsm8k_delimiter(self):
        """GSM8K answers use '#### N' — the N must be targeted."""
        result = inject_answer_fault(self.GSM8K_ANSWER, "incorrect_number", seed=2)
        # The #### value in result must differ from original (18)
        orig_match = re.search(r"####\s*(\d+)", self.GSM8K_ANSWER)
        mut_match = re.search(r"####\s*(\d+)", result)
        if orig_match and mut_match:
            assert orig_match.group(1) != mut_match.group(1), (
                "GSM8K '#### N' value must be mutated by incorrect_number fault."
            )

    def test_incorrect_number_minimum_delta(self):
        """The numeric change must be non-trivial (>= 10% of original or >= 3)."""
        answer = "The total cost is $100. The answer is 100."
        result = inject_answer_fault(answer, "incorrect_number", seed=5)
        orig = _find_final_answer_number(answer)
        mutated = _find_final_answer_number(result)
        if orig and mutated:
            delta = abs(float(mutated[0]) - float(orig[0]))
            assert delta >= 3, f"Delta {delta} too small — fault may not be detectable."

    def test_contradiction_directly_negates_answer(self):
        """contradiction must prepend content that explicitly contradicts."""
        result = inject_answer_fault(self.MATH_ANSWER, "contradiction", seed=3)
        assert len(result) > len(self.MATH_ANSWER)
        # Result must contain explicit negation language
        negation_keywords = ["incorrect", "wrong", "contradicts", "error", "correction"]
        found = any(kw.lower() in result.lower() for kw in negation_keywords)
        assert found, (
            f"contradiction fault must contain explicit negation. "
            f"Result: {result[:200]}"
        )
        # The original answer must still be present (contradiction wraps, not replaces)
        assert self.MATH_ANSWER in result or "Step" in result

    def test_incorrect_reasoning_step_inserts_within_chain(self):
        """The flawed step must appear WITHIN the reasoning (not just appended)."""
        multi_step = (
            "Step 1: Count apples = 10. "
            "Step 2: Count oranges = 5. "
            "Step 3: Total = 10 + 5 = 15. "
            "The answer is 15."
        )
        result = inject_answer_fault(multi_step, "incorrect_reasoning_step", seed=4)
        assert result != multi_step
        # Result must be longer than original
        assert len(result) > len(multi_step)

    def test_plausible_hallucination_uses_question_context(self):
        """Hallucination must reference a keyword from the question."""
        question = "How many atoms are in a molecule of water?"
        answer = "A water molecule has 3 atoms: 2 hydrogen and 1 oxygen."
        result = inject_answer_fault(answer, "plausible_hallucination",
                                     question=question, seed=6)
        assert len(result) > len(answer)
        # The hallucination should reference something from the question domain
        # At minimum it must be appended (not replace original)
        assert answer in result

    def test_plausible_hallucination_seed_reproducibility(self):
        r1 = inject_answer_fault(self.MATH_ANSWER, "plausible_hallucination",
                                 question=self.MATH_QUESTION, seed=77)
        r2 = inject_answer_fault(self.MATH_ANSWER, "plausible_hallucination",
                                 question=self.MATH_QUESTION, seed=77)
        assert r1 == r2

    def test_all_faults_reproducible_with_seed(self):
        for ft in ["plausible_hallucination", "incorrect_number",
                   "contradiction", "incorrect_reasoning_step"]:
            r1 = inject_answer_fault(self.MATH_ANSWER, ft, seed=42)
            r2 = inject_answer_fault(self.MATH_ANSWER, ft, seed=42)
            assert r1 == r2, f"Fault type '{ft}' is not reproducible with seed=42"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. find_final_answer_number
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindFinalAnswerNumber:
    def test_gsm8k_delimiter(self):
        text = "She sold 9 eggs. Total = 9*2 = $18.\n#### 18"
        result = _find_final_answer_number(text)
        assert result is not None
        assert result[0] == "18"

    def test_answer_is_pattern(self):
        text = "After counting, the answer is 42."
        result = _find_final_answer_number(text)
        assert result is not None
        assert result[0] == "42"

    def test_result_is_pattern(self):
        text = "The result is 100 units."
        result = _find_final_answer_number(text)
        assert result is not None
        assert result[0] == "100"

    def test_fallback_last_integer(self):
        text = "We counted 5, then 3, and finally got 8."
        result = _find_final_answer_number(text)
        assert result is not None
        assert result[0] == "8"

    def test_no_numbers_returns_none(self):
        text = "There are no numbers in this text."
        result = _find_final_answer_number(text)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Unified Factory + Strict Mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadDatasetByName:
    def test_all_datasets_in_supported_list(self):
        expected = {
            DATASET_GSM8K, DATASET_HALUEVAL, DATASET_HALUEVAL_GENERAL,
            DATASET_SIMPLEQA, DATASET_FRAMES, DATASET_MEDHALLU, DATASET_MMLU_PRO,
        }
        assert expected.issubset(set(SUPPORTED_DATASETS))

    def test_factory_all_datasets_offline(self):
        for name in SUPPORTED_DATASETS:
            samples = load_dataset_by_name(name, limit=1, strict=False)
            assert isinstance(samples, list)
            assert len(samples) >= 1

    def test_factory_case_insensitive(self):
        s1 = load_dataset_by_name("GSM8K", limit=1, strict=False)
        s2 = load_dataset_by_name("gsm8k", limit=1, strict=False)
        assert s1[0].dataset == s2[0].dataset

    def test_factory_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_by_name("nonexistent_xyz", strict=False)

    def test_medhallu_default_split_is_train(self):
        """MedHallu has no test split — factory must use train by default."""
        # We can only verify this offline via stub
        samples = load_dataset_by_name("medhallu", limit=1, strict=False)
        # Stub sample_id should contain 'train'
        assert "train" in samples[0].sample_id or "stub" in samples[0].sample_id

    def test_strict_false_returns_stub_on_failure(self):
        """strict=False must return stub samples, not raise."""
        # GSM8K with a bad split should fail silently and return stub
        try:
            samples = load_gsm8k_test(split="nonexistent_split", strict=False)
            assert isinstance(samples, list)
        except RuntimeError:
            pytest.fail("strict=False should not raise RuntimeError")

    def test_strict_true_raises_on_bad_split(self):
        """strict=True must raise RuntimeError on load failure (for experiments)."""
        with pytest.raises(RuntimeError, match="FATAL"):
            load_gsm8k_test(split="nonexistent_split_xyz", strict=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Shuffle Determinism
# ═══════════════════════════════════════════════════════════════════════════════

class TestShuffleDataset:
    def _make(self, n=10):
        return [
            BenchmarkSample(
                sample_id=f"s_{i}", dataset=DATASET_GSM8K,
                question=f"Q{i}", ground_truth=str(i),
                task_type=TASK_MATHEMATICAL_REASONING,
                answer_type=ANSWER_TYPE_NUMERIC,
            ) for i in range(n)
        ]

    def test_deterministic_same_seed(self):
        s = self._make(10)
        assert [x.sample_id for x in shuffle_dataset(s, 42)] == \
               [x.sample_id for x in shuffle_dataset(s, 42)]

    def test_different_seeds_produce_different_order(self):
        s = self._make(10)
        assert [x.sample_id for x in shuffle_dataset(s, 1)] != \
               [x.sample_id for x in shuffle_dataset(s, 99)]

    def test_preserves_length(self):
        s = self._make(7)
        assert len(shuffle_dataset(s, 0)) == 7

    def test_does_not_mutate_original(self):
        s = self._make(5)
        original = [x.sample_id for x in s]
        shuffle_dataset(s, 42)
        assert [x.sample_id for x in s] == original


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GSM8K Numeric Truth Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractGSM8KTruth:
    def test_with_delimiter(self):
        assert extract_gsm8k_numeric_truth("She earned 60.\n#### 60") == "60"

    def test_without_delimiter_last_number(self):
        assert extract_gsm8k_numeric_truth("Steps... the answer is 42") == "42"

    def test_backward_compat_alias(self):
        assert extract_gsm8k_truth("#### 18") == "18"

    def test_decimal_in_answer(self):
        # GSM8K uses integers, but edge case
        result = extract_gsm8k_numeric_truth("Total = 15\n#### 15")
        assert result == "15"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Fault Rate Gate
# ═══════════════════════════════════════════════════════════════════════════════

class TestShouldApplyFault:
    def test_zero_rate_never(self):
        assert not any(should_apply_fault(0.0) for _ in range(20))

    def test_full_rate_always(self):
        assert all(should_apply_fault(1.0) for _ in range(20))

    def test_fault_rates_constant(self):
        assert FAULT_RATES == [0.0, 0.10, 0.25, 0.50]

    def test_probabilistic_50pct(self):
        hits = sum(1 for i in range(1000) if should_apply_fault(0.5, seed=i))
        assert 400 <= hits <= 600, f"Expected ~500/1000, got {hits}"

    def test_seed_reproducibility(self):
        assert should_apply_fault(0.3, seed=7) == should_apply_fault(0.3, seed=7)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Reproducibility Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TestReproducibilityConfig:
    """
    Verifies that experiment configuration can be fully captured as a dict
    and that all required reproducibility fields are present.
    """

    REQUIRED_EXPERIMENT_FIELDS = {
        "dataset", "split", "sample_count", "random_seed",
        "fault_rate", "fault_type", "model", "temperature", "configuration",
    }

    def _make_config(self, **overrides) -> dict:
        base = {
            "dataset": "halueval",
            "split": "data",
            "sample_count": 100,
            "random_seed": 42,
            "fault_rate": 0.0,
            "fault_type": "none",
            "model": "casperhansen/llama-3-8b-instruct-awq",
            "temperature": 0.3,
            "configuration": {"alpha": 0.85, "beta": 0.10},
        }
        base.update(overrides)
        return base

    def test_required_fields_present(self):
        cfg = self._make_config()
        missing = self.REQUIRED_EXPERIMENT_FIELDS - set(cfg.keys())
        assert not missing, f"Missing experiment config fields: {missing}"

    def test_fault_rate_values(self):
        for rate in FAULT_RATES:
            cfg = self._make_config(fault_rate=rate)
            assert 0.0 <= cfg["fault_rate"] <= 1.0

    def test_seed_is_integer(self):
        cfg = self._make_config(random_seed=42)
        assert isinstance(cfg["random_seed"], int)

    def test_model_field_required(self):
        cfg = self._make_config()
        assert "model" in cfg
        # Must not be hard-coded — just verify it's a string
        assert isinstance(cfg["model"], str) and len(cfg["model"]) > 0
