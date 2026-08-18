"""
src/risk_gate — Deterministic Risk Gate for the AHC pipeline.

Public API:
    RiskLevel       — LOW / MEDIUM / HIGH enum.
    RiskFeatures    — dataclass of the five risk signals.
    RiskGateConfig  — weights, thresholds, fallback configuration.
    RiskGate        — stateless scoring and classification engine.
    assess_claims   — convenience function for batch assessment.
    FEATURE_NAMES   — canonical tuple of recognised feature names.
    BLOCKED_FIELDS  — evaluation-only field names that must never be used.
"""

from src.risk_gate.features import (
    BLOCKED_FIELDS,
    FEATURE_NAMES,
    RiskFeatures,
    RiskGateConfig,
    RiskLevel,
)
from src.risk_gate.gate import RiskGate, assess_claims

__all__ = [
    "BLOCKED_FIELDS",
    "FEATURE_NAMES",
    "RiskFeatures",
    "RiskGateConfig",
    "RiskGate",
    "RiskLevel",
    "assess_claims",
]
