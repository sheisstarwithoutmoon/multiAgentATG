"""
Risk Gate consumes pre-computed RiskFeatures and produces a normalised risk score and discrete risk level for each claim.
It does NOT generate, retrieve, or infer any signal values.

Risk equation:
R = w1 * uncertainty + w2 * disagreement + w3 * (1 - agent_trust) + w4 * claim_complexity + w5 * (1 - evidence_support)
where R is normalised to [0, 1] by dividing by the sum of weights.

Classification:
    R < low_threshold   → LOW
    R < high_threshold  → MEDIUM
    R >= high_threshold → HIGH
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.risk_gate.features import (
    BLOCKED_FIELDS,
    FEATURE_NAMES,
    RiskFeatures,
    RiskGateConfig,
    RiskLevel,
)

# Features whose risk contribution is *inverted* (higher value = safer).
_INVERTED_FEATURES: frozenset[str] = frozenset({"agent_trust", "evidence_support"})

class RiskGate:
    def __init__(self, config: Optional[RiskGateConfig] = None) -> None:
        self._config = config if config is not None else RiskGateConfig()

    @property
    def config(self) -> RiskGateConfig:
        return self._config
    
    def compute_score(self, features: RiskFeatures) -> float:
        weights = self._config.weights
        fallback = self._config.unavailable_value
        weight_sum = sum(weights.values())

        weighted_total = 0.0
        for name in FEATURE_NAMES:
            raw = getattr(features, name)
            value = raw if raw is not None else fallback

            # Invert "safer-is-higher" features so that the contribution
            # increases with risk.
            if name in _INVERTED_FEATURES:
                value = 1.0 - value

            weighted_total += weights[name] * value

        score = weighted_total / weight_sum
        # Clamp to [0, 1] as a safety net (should already be in range
        # given validated inputs, but defensive programming).
        return max(0.0, min(1.0, score))


    def classify(self, score: float) -> RiskLevel:
        """
        Thresholds (configurable):
            score < low_threshold   → LOW
            score < high_threshold  → MEDIUM
            score >= high_threshold → HIGH
        """
        low_upper, high_lower = self._config.thresholds
        if score < low_upper:
            return RiskLevel.LOW
        if score < high_lower:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def assess_claim(
        self,
        claim: str,
        features: Optional[RiskFeatures] = None,
    ) -> Dict[str, Any]:
        """
        Returns
        dict
            {
                "claim": str,
                "score": float,          # backward compat with router
                "risk_score": float,     # canonical name
                "risk_level": str,       # "LOW" / "MEDIUM" / "HIGH"
                "features": {…},         # raw feature values (incl. None)
                "signals_available": [str, …],
                "signals_missing": [str, …],
            }
        """
        if features is None:
            features = RiskFeatures()

        score = self.compute_score(features)
        level = self.classify(score)

        return {
            "claim": claim,
            "score": score,               # router reads this key
            "risk_score": score,           # canonical structured key
            "risk_level": level.value,     # str, not enum
            "features": features.to_dict(),
            "signals_available": features.available_signals(),
            "signals_missing": features.missing_signals(),
        }

    def assess_claims(
        self,
        claims: List[str],
        features_list: Optional[List[RiskFeatures]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Assess a batch of claims.
        Returns
        -------
        dict
            ``{"claim_risks": [<assessment>, …]}``
            Compatible with ``RoutingState["claim_risks"]``.

        Raises
        ------
        ValueError
            If ``features_list`` is provided but its length does not
            match ``claims``.
        """
        if features_list is not None and len(features_list) != len(claims):
            raise ValueError(
                f"features_list length ({len(features_list)}) must match "
                f"claims length ({len(claims)})"
            )

        results: List[Dict[str, Any]] = []
        for i, claim in enumerate(claims):
            feats = features_list[i] if features_list is not None else None
            results.append(self.assess_claim(claim, feats))

        return {"claim_risks": results}

def assess_claims(
    claims: List[str],
    features_list: Optional[List[RiskFeatures]] = None,
    config: Optional[RiskGateConfig] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function — create a gate and assess claims in one call."""
    gate = RiskGate(config)
    return gate.assess_claims(claims, features_list)
