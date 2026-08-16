"""
src/data/__init__.py
Public API for the data package.
"""
from src.data.benchmark_schema import (
    BenchmarkSample,
    PromptInput,
    EVAL_ONLY_FIELDS,
    VALID_TASK_TYPES,
    VALID_ANSWER_TYPES,
    ANSWER_TYPE_NUMERIC,
    ANSWER_TYPE_CONTAINS,
    ANSWER_TYPE_EXACT_MATCH,
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
)
from src.data.loader import (
    load_gsm8k_test,
    load_halueval_qa,
    load_halueval_general,
    load_simpleqa,
    load_frames,
    load_medhallu,
    load_mmlu_pro,
    load_dataset_by_name,
    shuffle_dataset,
    extract_gsm8k_numeric_truth,
    extract_gsm8k_truth,
    run_smoke_test,
    SUPPORTED_DATASETS,
)
from src.data.injection import (
    inject_adversarial_mutation,
    inject_answer_fault,
    should_apply_fault,
    FAULT_TYPES,
    FAULT_RATES,
    extract_gsm8k_truth as injection_extract_gsm8k_truth,
)
from src.data.evaluator import (
    EvaluationResult,
    evaluate_prediction,
    get_supported_answer_types,
)

__all__ = [
    # Schema
    "BenchmarkSample",
    "PromptInput",
    "EVAL_ONLY_FIELDS",
    "VALID_TASK_TYPES",
    "VALID_ANSWER_TYPES",
    "ANSWER_TYPE_NUMERIC", "ANSWER_TYPE_CONTAINS",
    "ANSWER_TYPE_EXACT_MATCH", "ANSWER_TYPE_BINARY_LABEL",
    "ANSWER_TYPE_LETTER_CHOICE",
    "TASK_MATHEMATICAL_REASONING", "TASK_FACTUAL_QA",
    "TASK_HALLUCINATION_DETECTION", "TASK_MULTIPLE_CHOICE",
    "TASK_MULTI_HOP_REASONING",
    "DATASET_GSM8K", "DATASET_HALUEVAL", "DATASET_HALUEVAL_GENERAL",
    "DATASET_SIMPLEQA", "DATASET_FRAMES", "DATASET_MEDHALLU", "DATASET_MMLU_PRO",
    # Loaders
    "load_gsm8k_test", "load_halueval_qa", "load_halueval_general",
    "load_simpleqa", "load_frames", "load_medhallu", "load_mmlu_pro",
    "load_dataset_by_name", "shuffle_dataset",
    "extract_gsm8k_numeric_truth", "extract_gsm8k_truth",
    "run_smoke_test", "SUPPORTED_DATASETS",
    # Injection
    "inject_adversarial_mutation", "inject_answer_fault",
    "should_apply_fault", "FAULT_TYPES", "FAULT_RATES",
    # Evaluation
    "EvaluationResult", "evaluate_prediction", "get_supported_answer_types",
]
