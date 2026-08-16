"""
src/data/benchmark_schema.py

Normalized sample schema for all AHC benchmarks.

VERIFIED real schemas (confirmed via direct HuggingFace load + inspection):
  GSM8K         openai/gsm8k (main), test — keys: question, answer
  HaluEval QA   pminervini/HaluEval (qa), data — keys: knowledge, question,
                  right_answer, hallucinated_answer
  HaluEval Gen  pminervini/HaluEval (general), data — keys: ID, user_query,
                  chatgpt_response, hallucination, hallucination_spans
  SimpleQA      basicv8vc/SimpleQA, test — keys: metadata (JSON), problem, answer
  FRAMES        google/frames-benchmark, test — keys: Prompt, Answer,
                  reasoning_types, wiki_links, wikipedia_link_N
  MedHallu      UTAustin-AIHealth/MedHallu (pqa_labeled/pqa_artificial), train
                  keys: Question, Knowledge (List[str]), Ground Truth,
                  Difficulty Level, Hallucinated Answer, Category of Hallucination
  MMLU-Pro      TIGER-Lab/MMLU-Pro, test — keys: question_id, question,
                  options (List[str], 9-10 items), answer (letter A-J),
                  answer_index (int, 0-based), cot_content, category, src

Design principles:
  1. task_type and answer_type are EXPLICIT per dataset — no universal default.
  2. Ground-truth fields are physically separated and must NEVER appear in any
     prompt. Use prompt_safe_dict() exclusively for prompt construction.
  3. knowledge is NOT automatically injected into prompts. Use PromptInput to
     explicitly opt-in on a per-benchmark/per-config basis.
  4. metadata must never contain values that duplicate ground_truth, hallucinated_answer,
     reference_label, hallucination_spans, or answer_index.
  5. inject_adversarial_mutation ONLY modifies the question. Ground truth is
     never touched and the returned dict is not a BenchmarkSample.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ─── Task type constants (semantic categories) ────────────────────────────────
TASK_MATHEMATICAL_REASONING  = "mathematical_reasoning"
TASK_FACTUAL_QA              = "factual_qa"
TASK_HALLUCINATION_DETECTION = "hallucination_detection"
TASK_MULTIPLE_CHOICE         = "multiple_choice"
TASK_MULTI_HOP_REASONING     = "multi_hop_reasoning"

VALID_TASK_TYPES = frozenset({
    TASK_MATHEMATICAL_REASONING,
    TASK_FACTUAL_QA,
    TASK_HALLUCINATION_DETECTION,
    TASK_MULTIPLE_CHOICE,
    TASK_MULTI_HOP_REASONING,
})

# ─── Answer type constants (evaluation mechanics) ────────────────────────────
ANSWER_TYPE_NUMERIC      = "numeric"       # extract final number, compare
ANSWER_TYPE_EXACT_MATCH  = "exact_match"   # normalized string equality
ANSWER_TYPE_CONTAINS     = "contains"      # ground truth ⊆ predicted text
ANSWER_TYPE_BINARY_LABEL = "binary_label"  # yes / no
ANSWER_TYPE_LETTER_CHOICE = "letter_choice" # A/B/C … J (multiple choice)

VALID_ANSWER_TYPES = frozenset({
    ANSWER_TYPE_NUMERIC,
    ANSWER_TYPE_EXACT_MATCH,
    ANSWER_TYPE_CONTAINS,
    ANSWER_TYPE_BINARY_LABEL,
    ANSWER_TYPE_LETTER_CHOICE,
})

# ─── Dataset name constants ───────────────────────────────────────────────────
DATASET_GSM8K           = "gsm8k"
DATASET_HALUEVAL        = "halueval"
DATASET_HALUEVAL_GENERAL = "halueval_general"
DATASET_SIMPLEQA        = "simpleqa"
DATASET_FRAMES          = "frames"
DATASET_MEDHALLU        = "medhallu"
DATASET_MMLU_PRO        = "mmlu_pro"

# ─── Evaluation-only fields (NEVER allowed in prompts) ───────────────────────
# These are the field names that are forbidden from any prompt construction.
# The set is a single source of truth — all prompt-safety checks reference it.
EVAL_ONLY_FIELDS: frozenset = frozenset({
    "ground_truth",
    "hallucinated_answer",
    "reference_label",
    "hallucination_spans",
    "answer_index",
})


# ─── Prompt Input Interface ───────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptInput:
    """
    The ONLY data structure that may be passed to prompt construction.

    Prompt construction code MUST accept PromptInput, not BenchmarkSample.
    This makes it structurally impossible to accidentally include ground-truth
    fields in a prompt, since PromptInput simply does not have them.

    Fields
    ------
    question        : The query text for the model. Never contains ground truth.
    knowledge       : Supporting context, included ONLY when the benchmark
                      configuration explicitly sets include_knowledge=True.
                      When include_knowledge is False (default), this is None
                      even if the underlying sample has knowledge.
    options         : Answer choices for multiple-choice benchmarks.
                      Always None for non-MC tasks.
    sample_id       : Identifier for logging / tracing only.
    dataset         : Dataset name constant for logging / tracing only.
    task_type       : Semantic task category (TASK_* constant).
    answer_type     : Evaluation mechanics (ANSWER_TYPE_* constant).
    reasoning_types : FRAMES reasoning type tag (logging only).

    WHAT IS DELIBERATELY ABSENT:
      ground_truth, hallucinated_answer, reference_label,
      hallucination_spans, answer_index, metadata, category,
      difficulty, wiki_links.

    These are absent by design — they must not reach prompt construction.
    """
    question: str
    sample_id: str
    dataset: str
    task_type: str
    answer_type: str
    knowledge: Optional[str] = None      # only set when config opts in
    options: Optional[List[str]] = None  # MC benchmarks only
    reasoning_types: Optional[str] = None


# ─── BenchmarkSample ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkSample:
    """
    Unified, research-grade representation of a single benchmark item.

    GROUND TRUTH ISOLATION GUARANTEE
    ---------------------------------
    Fields in EVAL_ONLY_FIELDS:
      ground_truth, hallucinated_answer, reference_label,
      hallucination_spans, answer_index

    must NEVER appear in any prompt. The methods to_prompt_input() and
    prompt_safe_dict() enforce this structurally.

    KNOWLEDGE EXPOSURE POLICY
    --------------------------
    Knowledge is NOT automatically included in prompts. Call
    to_prompt_input(include_knowledge=True) only when the benchmark
    configuration explicitly requires it. Default is False.

    METADATA POLICY
    ---------------
    metadata must not contain any value that duplicates an EVAL_ONLY_FIELDS
    value. This is checked at construction time in __post_init__.

    FAULT INJECTION POLICY
    ----------------------
    inject_adversarial_mutation modifies only the returned dict's 'question'
    field. The BenchmarkSample itself is never mutated. Ground truth is never
    modified by any injection function.
    """

    sample_id:   str
    dataset:     str
    question:    str
    ground_truth: str
    task_type:   str
    answer_type: str

    # ── Optional fields ────────────────────────────────────────────────────────
    knowledge:           Optional[str]       = None
    hallucinated_answer: Optional[str]       = None
    reference_label:     Optional[str]       = None
    hallucination_spans: Optional[List[str]] = None
    category:            Optional[str]       = None
    difficulty:          Optional[str]       = None
    options:             Optional[List[str]] = None
    answer_index:        Optional[int]       = None
    reasoning_types:     Optional[str]       = None
    wiki_links:          Optional[List[str]] = None
    metadata:            Dict[str, Any]      = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ── Validate task_type and answer_type are explicit, known values ──────
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type {self.task_type!r}. "
                f"Must be one of {sorted(VALID_TASK_TYPES)}"
            )
        if self.answer_type not in VALID_ANSWER_TYPES:
            raise ValueError(
                f"Invalid answer_type {self.answer_type!r}. "
                f"Must be one of {sorted(VALID_ANSWER_TYPES)}"
            )

        # ── Check metadata does not duplicate eval-only values ─────────────────
        self._validate_metadata_no_gt_values()

    def _validate_metadata_no_gt_values(self) -> None:
        """
        Raises ValueError if metadata contains a value that is identical to
        any eval-only field. This catches the case where ground truth leaks
        through metadata under a different key name.
        """
        eval_values: set = set()
        for fname in EVAL_ONLY_FIELDS:
            v = getattr(self, fname, None)
            if v is None:
                continue
            if isinstance(v, str) and v.strip():
                eval_values.add(v.strip().lower())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item.strip():
                        eval_values.add(item.strip().lower())
            elif isinstance(v, int):
                eval_values.add(str(v))

        if not eval_values:
            return

        def _flatten_values(obj: Any) -> List[str]:
            if isinstance(obj, str):
                return [obj.strip().lower()]
            if isinstance(obj, (list, tuple)):
                result = []
                for x in obj:
                    result.extend(_flatten_values(x))
                return result
            if isinstance(obj, dict):
                result = []
                for v in obj.values():
                    result.extend(_flatten_values(v))
                return result
            if obj is not None:
                return [str(obj).strip().lower()]
            return []

        for meta_val_str in _flatten_values(self.metadata):
            if meta_val_str and meta_val_str in eval_values:
                raise ValueError(
                    f"[GroundTruthLeak] metadata contains a value that duplicates "
                    f"an eval-only field. Offending value: {meta_val_str!r}\n"
                    f"metadata keys: {list(self.metadata.keys())}\n"
                    "Do NOT store ground truth, hallucinated answers, reference labels, "
                    "or answer indices in metadata."
                )

    # ── Prompt construction interface ──────────────────────────────────────────

    def to_prompt_input(self, include_knowledge: bool = False) -> PromptInput:
        """
        Creates a PromptInput — the ONLY object that may be passed to prompt
        construction code.

        Args:
            include_knowledge: If True, includes the knowledge field.
                               Only set this when the benchmark configuration
                               explicitly requires knowledge grounding (e.g.,
                               HaluEval QA and MedHallu in grounded mode).
                               Default is False.

        Returns:
            PromptInput with no eval-only fields present.
        """
        return PromptInput(
            question=self.question,
            sample_id=self.sample_id,
            dataset=self.dataset,
            task_type=self.task_type,
            answer_type=self.answer_type,
            knowledge=self.knowledge if include_knowledge else None,
            options=self.options,  # safe: these are the question options, not the answer
            reasoning_types=self.reasoning_types,
        )

    def prompt_safe_dict(self, include_knowledge: bool = False) -> Dict[str, Any]:
        """
        Returns a dict representation of PromptInput.
        Convenience wrapper around to_prompt_input().

        SAFE to pass to prompt construction — no eval-only fields present.
        """
        from dataclasses import asdict as _asdict
        return _asdict(self.to_prompt_input(include_knowledge=include_knowledge))

    # ── Evaluation interface ───────────────────────────────────────────────────

    def get_eval_fields(self) -> Dict[str, Any]:
        """
        Returns all evaluation-only fields for use by the evaluator.
        Must NEVER be passed to prompt construction code.
        """
        result = {}
        for fname in EVAL_ONLY_FIELDS:
            v = getattr(self, fname, None)
            if v is not None:
                result[fname] = v
        return result

    def get_forbidden_fields(self) -> Dict[str, Any]:
        """Alias for get_eval_fields(), kept for test compatibility."""
        return self.get_eval_fields()

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Full serialisation including eval-only fields. For storage/evaluation only."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkSample":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        init_kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            init_kwargs.setdefault("metadata", {}).update(extra)
        return cls(**init_kwargs)
