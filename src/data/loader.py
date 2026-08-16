"""
src/data/loader.py

Unified, research-grade dataset loaders for all AHC benchmark datasets.

VERIFIED REAL SCHEMAS (confirmed via direct HuggingFace load + inspection):
---------------------------------------------------------------------------
GSM8K         openai/gsm8k, config="main", split="test"
              keys: question, answer
              answer format: chain-of-thought + "#### <int>" delimiter
              rows: 1319 (test), 7473 (train)

HaluEval QA   pminervini/HaluEval, config="qa", split="data"
              keys: knowledge, question, right_answer, hallucinated_answer
              rows: 10000
              TASK: given knowledge+question, determine whether an answer
              hallucinated. The "right_answer" is the factual answer.
              NOTE: this is NOT ordinary QA — see task_type documentation.

HaluEval Gen  pminervini/HaluEval, config="general", split="data"
              keys: ID, user_query, chatgpt_response, hallucination,
                    hallucination_spans
              hallucination: "yes"/"no" string
              TASK: binary hallucination detection on a given response

SimpleQA      basicv8vc/SimpleQA, split="test"
              keys: metadata (JSON string), problem, answer
              rows: 4326
              metadata dict contains: topic, answer_type, urls

FRAMES        google/frames-benchmark, split="test"
              keys: Unnamed: 0, Prompt, Answer, wikipedia_link_1..11+,
                    reasoning_types, wiki_links
              rows: 824
              TASK: multi-hop factual reasoning

MedHallu      UTAustin-AIHealth/MedHallu
              configs: pqa_labeled (1000), pqa_artificial (9000)
              split: train only (no test split exists)
              keys: Question, Knowledge (List[str]), Ground Truth,
                    Difficulty Level, Hallucinated Answer,
                    Category of Hallucination
              TASK: hallucination detection / factual grounding in medicine

MMLU-Pro      TIGER-Lab/MMLU-Pro, split="test"
              keys: question_id, question, options (List[str], len=10),
                    answer (letter A-J), answer_index (int), cot_content,
                    category, src
              rows: 12032

STUB POLICY:
------------
Stubs are ONLY used for offline unit tests. Any call from an experiment
must fail loudly if the dataset cannot be loaded. Use strict=True
(default) in experiment scripts. Stubs are only activated when
strict=False (for development/unit tests).
"""

import ast
import json
import os
import random
from typing import List, Optional

from src.data.benchmark_schema import (
    BenchmarkSample,
    PromptInput,
    ANSWER_TYPE_NUMERIC,
    ANSWER_TYPE_EXACT_MATCH,
    ANSWER_TYPE_CONTAINS,
    ANSWER_TYPE_BINARY_LABEL,
    ANSWER_TYPE_LETTER_CHOICE,
    TASK_MATHEMATICAL_REASONING,
    TASK_FACTUAL_QA,
    TASK_HALLUCINATION_DETECTION,
    TASK_MULTIPLE_CHOICE,
    TASK_MULTI_HOP_REASONING,
    DATASET_GSM8K,
    DATASET_HALUEVAL,
    DATASET_HALUEVAL_GENERAL,
    DATASET_SIMPLEQA,
    DATASET_FRAMES,
    DATASET_MEDHALLU,
    DATASET_MMLU_PRO,
    EVAL_ONLY_FIELDS,
)

# ─── Internal helpers ──────────────────────────────────────────────────────────

def _hf_load(path: str, name: Optional[str], split: str, strict: bool = True):
    """
    Thin wrapper around datasets.load_dataset.
    Raises DatasetLoadError (RuntimeError) when strict=True and loading fails.
    """
    from datasets import load_dataset
    token = os.getenv("HF_TOKEN", None)
    kwargs: dict = {"split": split}
    if token:
        kwargs["token"] = token
    try:
        if name:
            return load_dataset(path, name, **kwargs)
        return load_dataset(path, **kwargs)
    except Exception as e:
        if strict:
            raise RuntimeError(
                f"[Loader] FATAL: Could not load dataset '{path}' "
                f"(config={name!r}, split={split!r}): {e}\n"
                "Set strict=False only for unit tests / offline development."
            ) from e
        raise  # re-raise so callers can handle with stubs


def extract_gsm8k_numeric_truth(answer: str) -> str:
    """
    Extracts the final numeric answer from a GSM8K solution chain.
    The canonical delimiter is '#### <integer>'.
    Falls back to the last integer if the delimiter is absent.
    """
    import re
    if "#### " in answer:
        return answer.split("#### ")[-1].strip()
    numbers = re.findall(r"\b\d+\b", answer)
    return numbers[-1] if numbers else answer.strip()


# Backward-compatible alias used by run_experiment.py
def extract_gsm8k_truth(answer: str) -> str:
    return extract_gsm8k_numeric_truth(answer)


# ─── Individual loaders ───────────────────────────────────────────────────────

def load_gsm8k_test(
    split: str = "test",
    limit: Optional[int] = None,
    strict: bool = True,
) -> List[BenchmarkSample]:
    """
    Loads GSM8K (openai/gsm8k, config="main").

    Real schema: keys = [question, answer]
    Answer format: chain-of-thought reasoning + '#### <int>' at end.
    Ground truth = numeric integer extracted from answer chain.

    Task type : mathematical_reasoning
    Answer type: numeric
    """
    # METADATA POLICY: raw_answer is the full reasoning chain, which as a
    # string contains the numeric ground truth as a substring. We must NOT
    # store it in metadata to avoid accidental leakage. Instead, we store
    # only a flag indicating the answer was chain-of-thought formatted.
    _STUB = [BenchmarkSample(
        sample_id="gsm8k_test_stub_00000",
        dataset=DATASET_GSM8K,
        question="Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much did she earn?",
        ground_truth="60",
        task_type=TASK_MATHEMATICAL_REASONING,
        answer_type=ANSWER_TYPE_NUMERIC,
        metadata={"has_cot": True, "stub": True},
    )]

    try:
        dataset = _hf_load("openai/gsm8k", "main", split, strict=strict)
    except RuntimeError:
        raise
    except Exception:
        if not strict:
            return _STUB
        raise

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break
        gt = extract_gsm8k_numeric_truth(row["answer"])
        # Do NOT store raw_answer in metadata — it contains the GT as a substring.
        # Store only structural metadata that cannot carry GT values.
        samples.append(BenchmarkSample(
            sample_id=f"gsm8k_{split}_{i:05d}",
            dataset=DATASET_GSM8K,
            question=row["question"],
            ground_truth=gt,
            task_type=TASK_MATHEMATICAL_REASONING,
            answer_type=ANSWER_TYPE_NUMERIC,
            metadata={"has_cot": True},
        ))
    return samples


def load_halueval_qa(
    split: str = "data",
    limit: Optional[int] = None,
    strict: bool = True,
) -> List[BenchmarkSample]:
    """
    Loads HaluEval QA subset (pminervini/HaluEval, config="qa").

    VERIFIED SCHEMA: knowledge, question, right_answer, hallucinated_answer

    TASK CONTEXT: HaluEval QA is NOT ordinary factual QA. It is a
    hallucination detection/factual grounding benchmark. Each sample provides:
      - knowledge: a supporting passage (the factual grounding)
      - question:  the question asked
      - right_answer: the factual answer
      - hallucinated_answer: a plausible but incorrect answer (label = wrong)

    For the AHC pipeline this is used to test whether the system:
      (a) produces the right_answer rather than the hallucinated_answer, and
      (b) correctly identifies hallucinated claims.

    Ground truth = right_answer.
    Knowledge passage is provided as context (NOT injected into prompts
    automatically; the pipeline must decide whether to use it).

    Task type : hallucination_detection
    Answer type: contains
    """
    _STUB = [BenchmarkSample(
        sample_id="halueval_qa_data_stub_00000",
        dataset=DATASET_HALUEVAL,
        question="Who was the director of the movie Inception?",
        ground_truth="Christopher Nolan",
        task_type=TASK_HALLUCINATION_DETECTION,
        answer_type=ANSWER_TYPE_CONTAINS,
        knowledge="Inception is a 2010 science fiction film written and directed by Christopher Nolan.",
        hallucinated_answer="Steven Spielberg directed Inception.",
        metadata={"stub": True},
    )]

    try:
        dataset = _hf_load("pminervini/HaluEval", "qa", split, strict=strict)
    except RuntimeError:
        raise
    except Exception:
        if not strict:
            return _STUB
        raise

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break
        samples.append(BenchmarkSample(
            sample_id=f"halueval_qa_{split}_{i:05d}",
            dataset=DATASET_HALUEVAL,
            question=row["question"],
            ground_truth=row["right_answer"],
            task_type=TASK_HALLUCINATION_DETECTION,
            answer_type=ANSWER_TYPE_CONTAINS,
            knowledge=row.get("knowledge") or None,
            hallucinated_answer=row.get("hallucinated_answer") or None,
        ))
    return samples


def load_halueval_general(
    split: str = "data",
    limit: Optional[int] = None,
    strict: bool = True,
) -> List[BenchmarkSample]:
    """
    Loads HaluEval General subset (pminervini/HaluEval, config="general").

    VERIFIED SCHEMA: ID, user_query, chatgpt_response, hallucination (yes/no),
                     hallucination_spans (List[str])

    TASK CONTEXT: Binary hallucination detection. Given a (query, response)
    pair, determine whether the response contains hallucinations.

    Ground truth = hallucination binary label ("yes" or "no").
    The chatgpt_response is NOT sent to the model being evaluated — it is
    the entity being evaluated FOR hallucination.

    Task type : hallucination_detection
    Answer type: binary_label
    """
    # METADATA POLICY: chatgpt_response is already stored in the knowledge
    # field. Do NOT duplicate it in metadata — that would store the
    # response-under-evaluation in two places, increasing leakage surface area.
    _STUB = [BenchmarkSample(
        sample_id="halueval_general_data_stub_00000",
        dataset=DATASET_HALUEVAL_GENERAL,
        question="Produce a list of common words in the English language.",
        ground_truth="no",
        task_type=TASK_HALLUCINATION_DETECTION,
        answer_type=ANSWER_TYPE_BINARY_LABEL,
        reference_label="no",
        # knowledge stores the chatgpt_response — the text being assessed for
        # hallucination. It is NOT automatically injected into prompts; the
        # pipeline must call to_prompt_input(include_knowledge=True) explicitly.
        knowledge="the, a, and, to, in, that, is, was, for, on, are, with, as",
        metadata={"stub": True},
    )]

    try:
        dataset = _hf_load("pminervini/HaluEval", "general", split, strict=strict)
    except RuntimeError:
        raise
    except Exception:
        if not strict:
            return _STUB
        raise

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break
        label = row["hallucination"].strip().lower()  # "yes" or "no"
        spans = row.get("hallucination_spans", [])
        if isinstance(spans, str):
            try:
                spans = ast.literal_eval(spans)
            except Exception:
                spans = [spans] if spans else []
        samples.append(BenchmarkSample(
            sample_id=f"halueval_general_{split}_{i:05d}",
            dataset=DATASET_HALUEVAL_GENERAL,
            question=row["user_query"],
            ground_truth=label,
            task_type=TASK_HALLUCINATION_DETECTION,
            answer_type=ANSWER_TYPE_BINARY_LABEL,
            reference_label=label,
            hallucination_spans=spans or None,
            # The chatgpt_response is the text being assessed — store in
            # knowledge so downstream code can access it without it appearing
            # in forbidden fields.
            knowledge=row.get("chatgpt_response"),
            metadata={"original_id": row.get("ID")},
        ))
    return samples


def load_simpleqa(
    split: str = "test",
    limit: Optional[int] = None,
    strict: bool = True,
) -> List[BenchmarkSample]:
    """
    Loads SimpleQA (basicv8vc/SimpleQA).

    VERIFIED SCHEMA: metadata (JSON string), problem, answer
    metadata dict contains: topic, answer_type, urls

    Task type : factual_qa
    Answer type: exact_match
    """
    _STUB = [BenchmarkSample(
        sample_id="simpleqa_test_stub_00000",
        dataset=DATASET_SIMPLEQA,
        question="In which year was the Eiffel Tower completed?",
        ground_truth="1889",
        task_type=TASK_FACTUAL_QA,
        answer_type=ANSWER_TYPE_EXACT_MATCH,
        metadata={"stub": True, "topic": "History", "answer_type_meta": "Date"},
    )]

    try:
        dataset = _hf_load("basicv8vc/SimpleQA", None, split, strict=strict)
    except RuntimeError:
        raise
    except Exception:
        if not strict:
            return _STUB
        raise

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break
        # Parse metadata JSON string
        meta: dict = {}
        try:
            meta = ast.literal_eval(row.get("metadata", "{}"))
        except Exception:
            try:
                meta = json.loads(row.get("metadata", "{}"))
            except Exception:
                meta = {"raw_metadata": str(row.get("metadata", ""))}

        samples.append(BenchmarkSample(
            sample_id=f"simpleqa_{split}_{i:05d}",
            dataset=DATASET_SIMPLEQA,
            question=row["problem"],
            ground_truth=row["answer"],
            task_type=TASK_FACTUAL_QA,
            answer_type=ANSWER_TYPE_EXACT_MATCH,
            category=meta.get("topic"),
            metadata={
                "answer_type_meta": meta.get("answer_type"),
                "urls": meta.get("urls", []),
            },
        ))
    return samples


def load_frames(
    split: str = "test",
    limit: Optional[int] = None,
    strict: bool = True,
) -> List[BenchmarkSample]:
    """
    Loads FRAMES (google/frames-benchmark).

    VERIFIED SCHEMA: Unnamed: 0, Prompt, Answer, wikipedia_link_1..11+,
                     reasoning_types, wiki_links

    Task type : multi_hop_reasoning
    Answer type: contains
    """
    _STUB = [BenchmarkSample(
        sample_id="frames_test_stub_00000",
        dataset=DATASET_FRAMES,
        question="Who was the first person to walk on the Moon, and in what year?",
        ground_truth="Neil Armstrong in 1969",
        task_type=TASK_MULTI_HOP_REASONING,
        answer_type=ANSWER_TYPE_CONTAINS,
        reasoning_types="Multiple constraints",
        metadata={"stub": True},
    )]

    try:
        dataset = _hf_load("google/frames-benchmark", None, split, strict=strict)
    except RuntimeError:
        raise
    except Exception:
        if not strict:
            return _STUB
        raise

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break

        # wiki_links is stored as a stringified list
        wiki_links: List[str] = []
        raw_wl = row.get("wiki_links", "")
        if isinstance(raw_wl, list):
            wiki_links = raw_wl
        elif isinstance(raw_wl, str) and raw_wl:
            try:
                wiki_links = ast.literal_eval(raw_wl)
            except Exception:
                wiki_links = [raw_wl]

        samples.append(BenchmarkSample(
            sample_id=f"frames_{split}_{i:05d}",
            dataset=DATASET_FRAMES,
            question=row["Prompt"],
            ground_truth=row["Answer"],
            task_type=TASK_MULTI_HOP_REASONING,
            answer_type=ANSWER_TYPE_CONTAINS,
            reasoning_types=row.get("reasoning_types"),
            wiki_links=wiki_links or None,
            metadata={"original_index": row.get("Unnamed: 0")},
        ))
    return samples


def load_medhallu(
    config: str = "pqa_labeled",
    split: str = "train",
    limit: Optional[int] = None,
    strict: bool = True,
) -> List[BenchmarkSample]:
    """
    Loads MedHallu (UTAustin-AIHealth/MedHallu).

    VERIFIED SCHEMA:
      - Question (str)
      - Knowledge (List[str])  — list of supporting passages
      - Ground Truth (str)     — the factual/correct answer
      - Difficulty Level (str) — easy/medium/hard
      - Hallucinated Answer (str)
      - Category of Hallucination (str):
          Misinterpretation of Question, Incomplete Information,
          Mechanism and Pathway Misattribution,
          Methodological and Evidence Fabrication

    Available configs: pqa_labeled (1000 rows), pqa_artificial (9000 rows)
    Available splits: train ONLY (no test split in this dataset)

    TASK CONTEXT: Hallucination detection in biomedical text. Given a medical
    question and knowledge passages, determine whether a given answer is
    factually correct or hallucinated. The 'Ground Truth' is the correct
    reference answer; 'Hallucinated Answer' is the known-wrong foil.

    Task type : hallucination_detection
    Answer type: contains
    """
    _STUB = [BenchmarkSample(
        sample_id="medhallu_train_stub_00000",
        dataset=DATASET_MEDHALLU,
        question="Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
        ground_truth=(
            "Results depicted mitochondrial dynamics in vivo as PCD progresses "
            "within the lace plant."
        ),
        task_type=TASK_HALLUCINATION_DETECTION,
        answer_type=ANSWER_TYPE_CONTAINS,
        knowledge="Programmed cell death (PCD) is the regulated death of cells within an organism.",
        hallucinated_answer=(
            "Mitochondria regulate the formation of perforations in lace plant leaves "
            "through the modulation of calcium channels."
        ),
        category="Mechanism and Pathway Misattribution",
        difficulty="medium",
        metadata={"stub": True, "config": config},
    )]

    try:
        dataset = _hf_load("UTAustin-AIHealth/MedHallu", config, split, strict=strict)
    except RuntimeError:
        raise
    except Exception:
        if not strict:
            return _STUB
        raise

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break

        # Knowledge is a List[str] — join for downstream use
        raw_knowledge = row.get("Knowledge", [])
        if isinstance(raw_knowledge, list):
            knowledge_str = " ".join(str(k) for k in raw_knowledge)
        else:
            knowledge_str = str(raw_knowledge)

        samples.append(BenchmarkSample(
            sample_id=f"medhallu_{config}_{split}_{i:05d}",
            dataset=DATASET_MEDHALLU,
            question=row["Question"],
            ground_truth=row["Ground Truth"],
            task_type=TASK_HALLUCINATION_DETECTION,
            answer_type=ANSWER_TYPE_CONTAINS,
            knowledge=knowledge_str or None,
            hallucinated_answer=row.get("Hallucinated Answer") or None,
            category=row.get("Category of Hallucination") or None,
            difficulty=row.get("Difficulty Level") or None,
            metadata={"config": config},
        ))
    return samples


def load_mmlu_pro(
    split: str = "test",
    limit: Optional[int] = None,
    category: Optional[str] = None,
    strict: bool = True,
) -> List[BenchmarkSample]:
    """
    Loads MMLU-Pro (TIGER-Lab/MMLU-Pro).

    VERIFIED SCHEMA:
      question_id (int), question (str), options (List[str], len=10),
      answer (str, letter A-J), answer_index (int, 0-based),
      cot_content (str), category (str), src (str)

    Ground truth = answer letter (e.g. "A", "F", "I").
    answer_index = 0-based integer index into options list.

    IMPORTANT: options are NOT included in ground_truth or reference_label;
    the correct option text can be derived as options[answer_index].

    Task type : multiple_choice
    Answer type: letter_choice
    """
    _STUB = [BenchmarkSample(
        sample_id="mmlu_pro_test_stub_00000",
        dataset=DATASET_MMLU_PRO,
        question="Which organelle is responsible for ATP production?",
        ground_truth="B",
        task_type=TASK_MULTIPLE_CHOICE,
        answer_type=ANSWER_TYPE_LETTER_CHOICE,
        options=["Ribosome", "Mitochondria", "Nucleus", "Golgi Apparatus",
                 "Endoplasmic Reticulum", "Lysosome", "Cytoskeleton",
                 "Vacuole", "Cell membrane", "Chloroplast"],
        answer_index=1,
        category="biology",
        metadata={"stub": True},
    )]

    try:
        dataset = _hf_load("TIGER-Lab/MMLU-Pro", None, split, strict=strict)
    except RuntimeError:
        raise
    except Exception:
        if not strict:
            return _STUB
        raise

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(dataset):
        if limit and len(samples) >= limit:
            break
        if category and row.get("category", "") != category:
            continue

        options = row.get("options", [])
        # options is a real List[str] from HF (verified)
        answer_letter = row["answer"]          # e.g. "A", "I"
        answer_idx = row["answer_index"]       # 0-based int (verified)

        samples.append(BenchmarkSample(
            sample_id=f"mmlu_pro_{split}_{i:05d}",
            dataset=DATASET_MMLU_PRO,
            question=row["question"],
            ground_truth=answer_letter,
            task_type=TASK_MULTIPLE_CHOICE,
            answer_type=ANSWER_TYPE_LETTER_CHOICE,
            options=options,
            answer_index=answer_idx,
            category=row.get("category"),
            difficulty=row.get("src"),
            metadata={"question_id": row.get("question_id"), "cot_content": row.get("cot_content", "")},
        ))
    return samples


# ─── Unified factory ──────────────────────────────────────────────────────────

_LOADER_REGISTRY = {
    DATASET_GSM8K:           load_gsm8k_test,
    DATASET_HALUEVAL:        load_halueval_qa,
    DATASET_HALUEVAL_GENERAL: load_halueval_general,
    DATASET_SIMPLEQA:        load_simpleqa,
    DATASET_FRAMES:          load_frames,
    DATASET_MEDHALLU:        load_medhallu,
    DATASET_MMLU_PRO:        load_mmlu_pro,
}

SUPPORTED_DATASETS = list(_LOADER_REGISTRY.keys())

# Default splits for datasets that don't use "test"
_DEFAULT_SPLITS = {
    DATASET_HALUEVAL:         "data",
    DATASET_HALUEVAL_GENERAL: "data",
    DATASET_MEDHALLU:         "train",  # MedHallu has train only
}


def load_dataset_by_name(
    name: str,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    strict: bool = True,
    **kwargs,
) -> List[BenchmarkSample]:
    """
    Unified dataset factory.

    Args:
        name   : One of SUPPORTED_DATASETS (case-insensitive).
        split  : Dataset split. If None, uses the dataset's canonical default.
        limit  : Optional cap on number of samples returned.
        strict : If True (default for experiments), raises on load failure.
                 If False (for unit tests), falls back to offline stubs.
        **kwargs: Forwarded to the dataset-specific loader
                 (e.g. category= for mmlu_pro, config= for medhallu).

    Returns:
        List[BenchmarkSample]

    Raises:
        ValueError  if name is not in SUPPORTED_DATASETS.
        RuntimeError if strict=True and the dataset cannot be loaded.
    """
    name_lower = name.lower().strip()
    if name_lower not in _LOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {SUPPORTED_DATASETS}"
        )

    resolved_split = split or _DEFAULT_SPLITS.get(name_lower, "test")
    loader_fn = _LOADER_REGISTRY[name_lower]

    return loader_fn(split=resolved_split, limit=limit, strict=strict, **kwargs)


def shuffle_dataset(
    samples: List[BenchmarkSample],
    seed: int = 42,
) -> List[BenchmarkSample]:
    """
    Deterministically shuffles samples without modifying the original list.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled


def run_smoke_test(n_per_dataset: int = 3, strict: bool = True) -> dict:
    """
    Loads n_per_dataset real samples from every supported benchmark and
    prints a structured report.  This is the REAL DATA smoke test.

    Args:
        n_per_dataset: Number of samples to load per dataset.
        strict: If True, raises on any load failure (required for experiments).

    Returns:
        Dict mapping dataset_name -> {"loaded": int, "error": str|None}
    """
    results = {}

    for name in SUPPORTED_DATASETS:
        print(f"\n{'='*60}")
        print(f"SMOKE TEST: {name.upper()}")
        print(f"{'='*60}")
        try:
            samples = load_dataset_by_name(name, limit=n_per_dataset, strict=strict)
            results[name] = {"loaded": len(samples), "error": None}
            for s in samples:
                print(f"  sample_id    : {s.sample_id}")
                print(f"  dataset      : {s.dataset}")
                print(f"  task_type    : {s.task_type}")
                print(f"  answer_type  : {s.answer_type}")
                print(f"  question     : {s.question[:120]}...")
                if s.knowledge:
                    print(f"  knowledge    : {s.knowledge[:100]}...")
                print(f"  ground_truth : {s.ground_truth[:100]}")
                if s.hallucinated_answer:
                    print(f"  hallucinated : {s.hallucinated_answer[:100]}")
                if s.reference_label:
                    print(f"  ref_label    : {s.reference_label}")
                if s.options:
                    print(f"  options      : {len(s.options)} options, correct={s.ground_truth}")
                if s.reasoning_types:
                    print(f"  reasoning    : {s.reasoning_types}")
                if s.category:
                    print(f"  category     : {s.category}")
                if s.difficulty:
                    print(f"  difficulty   : {s.difficulty}")
                print(f"  metadata     : {list(s.metadata.keys())}")
                print()
        except Exception as e:
            results[name] = {"loaded": 0, "error": str(e)}
            print(f"  ERROR: {e}")

    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    for name, r in results.items():
        status = f"OK ({r['loaded']} samples)" if r["error"] is None else f"FAILED: {r['error'][:80]}"
        print(f"  {name:<25} {status}")

    return results
