
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

class RiskLevel(Enum):
    """Discrete risk classification assigned by the Risk Gate."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# Blocked fields: evaluation/ground-truth data that must NEVER be used
#: Fields that belong to the evaluation harness, not to the risk model.
#: The Risk Gate must never accept or inspect these.
BLOCKED_FIELDS: frozenset[str] = frozenset({
    "ground_truth",
    "hidden_benchmark_labels",
    "answer_index",
    "hallucinated_answer",
    "reference_label",
    "hallucination_spans",
    "benchmark_evaluation",
    "correct_answer",
    "gold_answer",
    "gold_label",
    "expected_answer",
    "target_answer",
    "label",
})


#: The five recognised risk-signal names, in canonical order.
FEATURE_NAMES: tuple[str, ...] = (
    "uncertainty",
    "disagreement",
    "agent_trust",
    "claim_complexity",
    "evidence_support",
)


@dataclass(frozen=True)
class RiskFeatures:
    uncertainty: Optional[float] = None
    disagreement: Optional[float] = None
    agent_trust: Optional[float] = None
    claim_complexity: Optional[float] = None
    evidence_support: Optional[float] = None

    def __post_init__(self) -> None:
        for name in FEATURE_NAMES:
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Feature '{name}' must be a float in [0, 1] or None, "
                        f"got {type(value).__name__}"
                    )
                if not (0.0 <= value <= 1.0):
                    raise ValueError(
                        f"Feature '{name}' must be in [0, 1], got {value}"
                    )

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return a plain dict of feature values (including None)."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    def available_signals(self) -> list[str]:
        """Return names of features that have an observed (non-None) value."""
        return [name for name in FEATURE_NAMES if getattr(self, name) is not None]

    def missing_signals(self) -> list[str]:
        """Return names of features that are None (unavailable)."""
        return [name for name in FEATURE_NAMES if getattr(self, name) is None]


def _default_weights() -> Dict[str, float]:
    return {name: 0.20 for name in FEATURE_NAMES}


def _default_thresholds() -> tuple[float, float]:
    """Default thresholds: LOW < 0.33, MEDIUM < 0.66, HIGH >= 0.66."""
    return (0.33, 0.66)


@dataclass
class RiskGateConfig:
    """Configuration for the Risk Gate.

    Attributes:
        weights:
            Mapping from feature name to non-negative weight.
            INITIAL EXPERIMENTAL DEFAULTS: 0.20 each (equal).
            Must be tuned/validated on training or development data —
            not the test set.

        thresholds:
            (low_upper, high_lower).
            score < low_upper  → LOW
            low_upper <= score < high_lower → MEDIUM
            score >= high_lower → HIGH

        unavailable_value:
            Deterministic neutral placeholder substituted for any signal
            that is ``None``.  This is NOT a measured or modelled signal —
            it exists solely so the pipeline can execute before all
            upstream components supply real values.
            Default: 0.5 (neutral midpoint).
    """

    weights: Dict[str, float] = field(default_factory=_default_weights)
    thresholds: tuple[float, float] = field(default_factory=_default_thresholds)
    unavailable_value: float = 0.5

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name in FEATURE_NAMES:
            if name not in self.weights:
                raise ValueError(f"Missing weight for feature '{name}'")
        for name, w in self.weights.items():
            if name not in FEATURE_NAMES:
                raise ValueError(
                    f"Unknown feature '{name}' in weights. "
                    f"Recognised features: {FEATURE_NAMES}"
                )
            if not isinstance(w, (int, float)):
                raise TypeError(f"Weight for '{name}' must be numeric, got {type(w).__name__}")
            if w < 0:
                raise ValueError(f"Weight for '{name}' must be non-negative, got {w}")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError(f"Sum of weights must be > 0, got {total}")

        # thresholds
        low_upper, high_lower = self.thresholds
        if not (0 < low_upper < high_lower < 1):
            raise ValueError(
                f"Thresholds must satisfy 0 < low_upper < high_lower < 1, "
                f"got ({low_upper}, {high_lower})"
            )

        # unavailable_value
        if not (0.0 <= self.unavailable_value <= 1.0):
            raise ValueError(
                f"unavailable_value must be in [0, 1], got {self.unavailable_value}"
            )
