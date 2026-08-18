"""
tests/test_risk_gate.py

Comprehensive test suite for the Risk Gate component.

Covers:
    1.  LOW / MEDIUM / HIGH classification
    2.  Exact threshold boundary behaviour
    3.  Score always in [0, 1]
    4.  Invalid feature rejection
    5.  Weight validation
    6.  Deterministic repeated execution
    7.  No random module usage
    8.  Monotonicity of every signal
    9.  Ground-truth / evaluation-only field rejection
   10.  Multiple claims produce independent assessments
   11.  RoutingState compatibility
   12.  Exact weighted-score calculation
   13.  Trust inversion
   14.  Evidence inversion
   15.  Custom weights
   16.  Custom thresholds
   17.  Missing-signal fallback & signals_available tracking

Run with:
    cd c:\\Users\\iamva\\multi_agent_atg
    python -m pytest tests/test_risk_gate.py -v
"""

import inspect
import pytest
from typing import Dict, Any

from src.risk_gate.features import (
    BLOCKED_FIELDS,
    FEATURE_NAMES,
    RiskFeatures,
    RiskGateConfig,
    RiskLevel,
)
from src.risk_gate.gate import RiskGate


# =====================================================================
# Helpers
# =====================================================================

def _make_gate(**config_kwargs) -> RiskGate:
    """Create a RiskGate with optional config overrides."""
    if config_kwargs:
        return RiskGate(RiskGateConfig(**config_kwargs))
    return RiskGate()


def _assess_one(
    gate: RiskGate,
    features: RiskFeatures | None = None,
) -> Dict[str, Any]:
    """Assess a single dummy claim and return the assessment dict."""
    result = gate.assess_claim("test claim", features)
    return result


# =====================================================================
# 1–3. Basic classification: LOW, MEDIUM, HIGH
# =====================================================================

class TestBasicClassification:
    """LOW / MEDIUM / HIGH classification with default thresholds."""

    def test_low_risk(self):
        """Features that should produce LOW risk."""
        # All safe: low uncertainty, no disagreement, high trust,
        # simple claim, strong evidence.
        features = RiskFeatures(
            uncertainty=0.0,
            disagreement=0.0,
            agent_trust=1.0,
            claim_complexity=0.0,
            evidence_support=1.0,
        )
        gate = RiskGate()
        result = _assess_one(gate, features)
        assert result["risk_level"] == "LOW"
        assert result["risk_score"] < 0.33

    def test_high_risk(self):
        """Features that should produce HIGH risk."""
        # All risky: max uncertainty, max disagreement, zero trust,
        # max complexity, no evidence.
        features = RiskFeatures(
            uncertainty=1.0,
            disagreement=1.0,
            agent_trust=0.0,
            claim_complexity=1.0,
            evidence_support=0.0,
        )
        gate = RiskGate()
        result = _assess_one(gate, features)
        assert result["risk_level"] == "HIGH"
        assert result["risk_score"] >= 0.66

    def test_medium_risk_all_midpoint(self):
        """All features at 0.5 → score = 0.5 → MEDIUM."""
        features = RiskFeatures(
            uncertainty=0.5,
            disagreement=0.5,
            agent_trust=0.5,
            claim_complexity=0.5,
            evidence_support=0.5,
        )
        gate = RiskGate()
        result = _assess_one(gate, features)
        assert result["risk_level"] == "MEDIUM"
        assert 0.33 <= result["risk_score"] < 0.66


# =====================================================================
# 4. Exact threshold boundary behaviour
# =====================================================================

class TestThresholdBoundaries:
    """Verify boundary conditions at 0.33 and 0.66."""

    def _score_via_single_feature(self, value: float) -> Dict[str, Any]:
        """With equal weights and only uncertainty varying, the score
        equals the weighted average.  We set all features explicitly
        to control the exact score.

        For a score of exactly S, set all features to produce S:
            uncertainty = S
            disagreement = S
            agent_trust = 1 - S      (inverted)
            claim_complexity = S
            evidence_support = 1 - S  (inverted)
        """
        features = RiskFeatures(
            uncertainty=value,
            disagreement=value,
            agent_trust=1.0 - value,
            claim_complexity=value,
            evidence_support=1.0 - value,
        )
        gate = RiskGate()
        return _assess_one(gate, features)

    def test_just_below_low_threshold(self):
        """score = 0.32 → LOW."""
        result = self._score_via_single_feature(0.32)
        assert abs(result["risk_score"] - 0.32) < 1e-9
        assert result["risk_level"] == "LOW"

    def test_exactly_at_low_threshold(self):
        """score = 0.33 → MEDIUM (not LOW)."""
        result = self._score_via_single_feature(0.33)
        assert abs(result["risk_score"] - 0.33) < 1e-9
        assert result["risk_level"] == "MEDIUM"

    def test_mid_medium_range(self):
        """score = 0.50 → MEDIUM."""
        result = self._score_via_single_feature(0.50)
        assert abs(result["risk_score"] - 0.50) < 1e-9
        assert result["risk_level"] == "MEDIUM"

    def test_just_below_high_threshold(self):
        """score = 0.659 → MEDIUM."""
        result = self._score_via_single_feature(0.659)
        assert abs(result["risk_score"] - 0.659) < 1e-9
        assert result["risk_level"] == "MEDIUM"

    def test_exactly_at_high_threshold(self):
        """score = 0.66 → HIGH."""
        result = self._score_via_single_feature(0.66)
        assert abs(result["risk_score"] - 0.66) < 1e-9
        assert result["risk_level"] == "HIGH"


# =====================================================================
# 5. Score always in [0, 1]
# =====================================================================

class TestScoreBounds:
    """Score must always be in [0, 1] regardless of inputs."""

    def test_minimum_risk(self):
        features = RiskFeatures(
            uncertainty=0.0, disagreement=0.0, agent_trust=1.0,
            claim_complexity=0.0, evidence_support=1.0,
        )
        score = RiskGate().compute_score(features)
        assert 0.0 <= score <= 1.0
        assert score == 0.0

    def test_maximum_risk(self):
        features = RiskFeatures(
            uncertainty=1.0, disagreement=1.0, agent_trust=0.0,
            claim_complexity=1.0, evidence_support=0.0,
        )
        score = RiskGate().compute_score(features)
        assert 0.0 <= score <= 1.0
        assert score == 1.0

    def test_all_none_fallback(self):
        """All missing → fallback 0.5 → score 0.5."""
        score = RiskGate().compute_score(RiskFeatures())
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.5) < 1e-9

    @pytest.mark.parametrize("val", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_score_in_range_parametric(self, val):
        features = RiskFeatures(
            uncertainty=val, disagreement=val,
            agent_trust=1.0 - val, claim_complexity=val,
            evidence_support=1.0 - val,
        )
        score = RiskGate().compute_score(features)
        assert 0.0 <= score <= 1.0


# =====================================================================
# 6. Invalid feature rejection
# =====================================================================

class TestFeatureValidation:
    """Invalid feature values must be rejected."""

    def test_negative_value(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            RiskFeatures(uncertainty=-0.1)

    def test_value_above_one(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            RiskFeatures(uncertainty=1.5)

    def test_non_numeric_type(self):
        with pytest.raises(TypeError, match="must be a float"):
            RiskFeatures(uncertainty="high")  # type: ignore

    def test_boundary_zero_is_valid(self):
        f = RiskFeatures(uncertainty=0.0)
        assert f.uncertainty == 0.0

    def test_boundary_one_is_valid(self):
        f = RiskFeatures(uncertainty=1.0)
        assert f.uncertainty == 1.0


# =====================================================================
# 7. Weight validation
# =====================================================================

class TestWeightValidation:

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            RiskGateConfig(weights={
                "uncertainty": -0.1, "disagreement": 0.25,
                "agent_trust": 0.25, "claim_complexity": 0.25,
                "evidence_support": 0.25,
            })

    def test_all_zero_weights_rejected(self):
        with pytest.raises(ValueError, match="Sum of weights must be > 0"):
            RiskGateConfig(weights={name: 0.0 for name in FEATURE_NAMES})

    def test_missing_feature_weight(self):
        with pytest.raises(ValueError, match="Missing weight"):
            RiskGateConfig(weights={"uncertainty": 1.0})

    def test_unknown_feature_weight(self):
        weights = {name: 0.2 for name in FEATURE_NAMES}
        weights["unknown_signal"] = 0.1
        with pytest.raises(ValueError, match="Unknown feature"):
            RiskGateConfig(weights=weights)

    def test_valid_unequal_weights(self):
        config = RiskGateConfig(weights={
            "uncertainty": 0.40, "disagreement": 0.20,
            "agent_trust": 0.15, "claim_complexity": 0.15,
            "evidence_support": 0.10,
        })
        assert sum(config.weights.values()) == pytest.approx(1.0)


# =====================================================================
# 8. Deterministic repeated execution
# =====================================================================

class TestDeterminism:
    """The gate must produce identical results across repeated calls."""

    def test_repeated_calls_identical(self):
        gate = RiskGate()
        features = RiskFeatures(uncertainty=0.7, disagreement=0.3)
        results = [_assess_one(gate, features) for _ in range(100)]
        first = results[0]
        for r in results[1:]:
            assert r["risk_score"] == first["risk_score"]
            assert r["risk_level"] == first["risk_level"]

    def test_new_gate_instances_identical(self):
        features = RiskFeatures(claim_complexity=0.8)
        results = [_assess_one(RiskGate(), features) for _ in range(50)]
        scores = {r["risk_score"] for r in results}
        assert len(scores) == 1  # all identical


# =====================================================================
# 9. No random module usage
# =====================================================================

class TestNoRandom:
    """The Risk Gate source code must not use the random module."""

    def _get_source(self, module) -> str:
        return inspect.getsource(module)

    def test_gate_module_no_random_import(self):
        from src.risk_gate import gate as gate_module
        source = self._get_source(gate_module)
        assert "import random" not in source
        assert "from random" not in source

    def test_features_module_no_random_import(self):
        from src.risk_gate import features as features_module
        source = self._get_source(features_module)
        assert "import random" not in source
        assert "from random" not in source

    def test_gate_module_no_random_call(self):
        from src.risk_gate import gate as gate_module
        source = self._get_source(gate_module)
        assert "random.uniform" not in source
        assert "random.random" not in source
        assert "random.choice" not in source
        assert "random.randint" not in source


# =====================================================================
# 10–14. Monotonicity: each signal's directional effect
# =====================================================================

class TestMonotonicity:
    """Each signal should have a strictly monotonic effect on risk
    when all other features are held fixed."""

    def _score_with(self, **overrides) -> float:
        """Score with a single feature overridden, others at 0.5."""
        defaults = {
            "uncertainty": 0.5,
            "disagreement": 0.5,
            "agent_trust": 0.5,
            "claim_complexity": 0.5,
            "evidence_support": 0.5,
        }
        defaults.update(overrides)
        return RiskGate().compute_score(RiskFeatures(**defaults))

    def test_higher_uncertainty_never_reduces_risk(self):
        low = self._score_with(uncertainty=0.2)
        high = self._score_with(uncertainty=0.8)
        assert high > low

    def test_higher_disagreement_never_reduces_risk(self):
        low = self._score_with(disagreement=0.2)
        high = self._score_with(disagreement=0.8)
        assert high > low

    def test_higher_agent_trust_never_increases_risk(self):
        low_trust = self._score_with(agent_trust=0.2)
        high_trust = self._score_with(agent_trust=0.8)
        assert high_trust < low_trust

    def test_higher_complexity_never_reduces_risk(self):
        low = self._score_with(claim_complexity=0.2)
        high = self._score_with(claim_complexity=0.8)
        assert high > low

    def test_higher_evidence_support_never_increases_risk(self):
        low_evidence = self._score_with(evidence_support=0.2)
        high_evidence = self._score_with(evidence_support=0.8)
        assert high_evidence < low_evidence

    @pytest.mark.parametrize("feature,direction", [
        ("uncertainty", "risk_increases"),
        ("disagreement", "risk_increases"),
        ("agent_trust", "risk_decreases"),
        ("claim_complexity", "risk_increases"),
        ("evidence_support", "risk_decreases"),
    ])
    def test_monotonicity_sweep(self, feature, direction):
        """Sweep a single feature from 0 to 1 in steps and verify
        that the risk score moves monotonically in the expected direction."""
        values = [i / 10.0 for i in range(11)]
        scores = [self._score_with(**{feature: v}) for v in values]
        for i in range(len(scores) - 1):
            if direction == "risk_increases":
                assert scores[i + 1] >= scores[i], (
                    f"{feature}={values[i+1]} gave lower risk than {values[i]}"
                )
            else:
                assert scores[i + 1] <= scores[i], (
                    f"{feature}={values[i+1]} gave higher risk than {values[i]}"
                )


# =====================================================================
# 15. Ground-truth / evaluation-only fields
# =====================================================================

class TestGroundTruthBlocked:
    """The Risk Gate must never accept evaluation-only fields."""

    def test_blocked_fields_not_in_feature_names(self):
        """No blocked field appears in the recognised feature names."""
        for blocked in BLOCKED_FIELDS:
            assert blocked not in FEATURE_NAMES, (
                f"Blocked field '{blocked}' must not be a risk feature"
            )

    def test_risk_features_rejects_blocked_kwargs(self):
        """RiskFeatures constructor must reject blocked field names."""
        for blocked in BLOCKED_FIELDS:
            with pytest.raises(TypeError):
                RiskFeatures(**{blocked: 0.5})  # type: ignore


# =====================================================================
# 16. Multiple claims produce independent assessments
# =====================================================================

class TestMultipleClaims:
    """Each claim's assessment must be independent."""

    def test_different_features_different_scores(self):
        gate = RiskGate()
        claims = ["claim A", "claim B"]
        features_list = [
            RiskFeatures(uncertainty=0.1),
            RiskFeatures(uncertainty=0.9),
        ]
        result = gate.assess_claims(claims, features_list)
        risks = result["claim_risks"]
        assert len(risks) == 2
        assert risks[0]["risk_score"] != risks[1]["risk_score"]
        assert risks[0]["claim"] == "claim A"
        assert risks[1]["claim"] == "claim B"

    def test_same_features_same_scores(self):
        gate = RiskGate()
        features = RiskFeatures(uncertainty=0.4, disagreement=0.6)
        result = gate.assess_claims(
            ["x", "y", "z"],
            [features, features, features],
        )
        scores = [r["risk_score"] for r in result["claim_risks"]]
        assert scores[0] == scores[1] == scores[2]

    def test_length_mismatch_raises(self):
        gate = RiskGate()
        with pytest.raises(ValueError, match="length"):
            gate.assess_claims(["a", "b"], [RiskFeatures()])


# =====================================================================
# 17. RoutingState compatibility
# =====================================================================

class TestRoutingStateCompat:
    """Output must be compatible with RoutingState['claim_risks']
    and the existing adaptive_agent_router which reads cr['score']."""

    def test_output_has_claim_risks_key(self):
        gate = RiskGate()
        result = gate.assess_claims(["test"])
        assert "claim_risks" in result
        assert isinstance(result["claim_risks"], list)

    def test_each_risk_has_score_key(self):
        """The router reads cr['score'] — this must be present."""
        gate = RiskGate()
        result = gate.assess_claims(["claim 1", "claim 2"])
        for cr in result["claim_risks"]:
            assert "score" in cr
            assert "claim" in cr
            assert isinstance(cr["score"], float)

    def test_score_and_risk_score_identical(self):
        gate = RiskGate()
        result = gate.assess_claims(["x"])
        cr = result["claim_risks"][0]
        assert cr["score"] == cr["risk_score"]

    def test_risk_level_is_string(self):
        gate = RiskGate()
        result = gate.assess_claims(["x"])
        cr = result["claim_risks"][0]
        assert cr["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_features_dict_present(self):
        gate = RiskGate()
        result = gate.assess_claims(["x"])
        cr = result["claim_risks"][0]
        assert "features" in cr
        assert set(cr["features"].keys()) == set(FEATURE_NAMES)

    def test_signals_available_and_missing(self):
        """signals_available and signals_missing must be in output."""
        gate = RiskGate()
        features = RiskFeatures(uncertainty=0.5)
        result = gate.assess_claims(["x"], [features])
        cr = result["claim_risks"][0]
        assert "signals_available" in cr
        assert "signals_missing" in cr
        assert "uncertainty" in cr["signals_available"]
        assert "uncertainty" not in cr["signals_missing"]
        assert "disagreement" in cr["signals_missing"]


# =====================================================================
# 12. Exact weighted-score calculation
# =====================================================================

class TestExactScoreCalculation:
    """Verify the weighted formula produces the expected numeric result."""

    def test_equal_weights_all_provided(self):
        """With equal weights (0.20 each), manually compute expected score.

        uncertainty=0.8, disagreement=0.6, agent_trust=0.7,
        claim_complexity=0.4, evidence_support=0.9

        Risk contributions:
            0.20 * 0.8             = 0.16   (uncertainty)
            0.20 * 0.6             = 0.12   (disagreement)
            0.20 * (1 - 0.7)       = 0.06   (agent_trust inverted)
            0.20 * 0.4             = 0.08   (claim_complexity)
            0.20 * (1 - 0.9)       = 0.02   (evidence_support inverted)
        Total = 0.44, weight_sum = 1.0, score = 0.44
        """
        features = RiskFeatures(
            uncertainty=0.8,
            disagreement=0.6,
            agent_trust=0.7,
            claim_complexity=0.4,
            evidence_support=0.9,
        )
        score = RiskGate().compute_score(features)
        assert score == pytest.approx(0.44, abs=1e-9)

    def test_custom_weights_exact(self):
        """Custom weights with a known expected result.

        weights: unc=0.4, dis=0.3, trust=0.1, comp=0.1, evi=0.1
        features: unc=1.0, dis=0.0, trust=1.0, comp=0.5, evi=0.0

        Contributions:
            0.4 * 1.0         = 0.40
            0.3 * 0.0         = 0.00
            0.1 * (1 - 1.0)   = 0.00
            0.1 * 0.5         = 0.05
            0.1 * (1 - 0.0)   = 0.10
        Total = 0.55, weight_sum = 1.0, score = 0.55
        """
        config = RiskGateConfig(weights={
            "uncertainty": 0.4, "disagreement": 0.3,
            "agent_trust": 0.1, "claim_complexity": 0.1,
            "evidence_support": 0.1,
        })
        features = RiskFeatures(
            uncertainty=1.0, disagreement=0.0, agent_trust=1.0,
            claim_complexity=0.5, evidence_support=0.0,
        )
        score = RiskGate(config).compute_score(features)
        assert score == pytest.approx(0.55, abs=1e-9)

    def test_unequal_weight_sum_normalisation(self):
        """Weights that don't sum to 1 should still normalise correctly.

        weights: unc=2, dis=1, trust=1, comp=1, evi=1 (sum=6)
        features: all 0.5, inverted ones also 0.5 after 1-0.5

        Each contribution: w * 0.5
        Total = (2+1+1+1+1) * 0.5 = 3.0
        score = 3.0 / 6 = 0.5
        """
        config = RiskGateConfig(weights={
            "uncertainty": 2.0, "disagreement": 1.0,
            "agent_trust": 1.0, "claim_complexity": 1.0,
            "evidence_support": 1.0,
        })
        features = RiskFeatures(
            uncertainty=0.5, disagreement=0.5, agent_trust=0.5,
            claim_complexity=0.5, evidence_support=0.5,
        )
        score = RiskGate(config).compute_score(features)
        assert score == pytest.approx(0.5, abs=1e-9)


# =====================================================================
# 13. Trust inversion
# =====================================================================

class TestTrustInversion:

    def test_trust_zero_contributes_max_risk(self):
        """agent_trust=0 → contribution = 1.0 (max risk)."""
        gate = RiskGate()
        # Only agent_trust supplied, rest at fallback 0.5
        f_zero = RiskFeatures(agent_trust=0.0)
        f_one = RiskFeatures(agent_trust=1.0)
        assert gate.compute_score(f_zero) > gate.compute_score(f_one)

    def test_trust_one_contributes_zero_risk(self):
        """agent_trust=1 → contribution = 0.0 (no risk from trust)."""
        features = RiskFeatures(
            uncertainty=0.0, disagreement=0.0, agent_trust=1.0,
            claim_complexity=0.0, evidence_support=1.0,
        )
        score = RiskGate().compute_score(features)
        assert score == pytest.approx(0.0, abs=1e-9)


# =====================================================================
# 14. Evidence inversion
# =====================================================================

class TestEvidenceInversion:

    def test_evidence_zero_contributes_max_risk(self):
        """evidence_support=0 → contribution = 1.0 (max risk)."""
        gate = RiskGate()
        f_zero = RiskFeatures(evidence_support=0.0)
        f_one = RiskFeatures(evidence_support=1.0)
        assert gate.compute_score(f_zero) > gate.compute_score(f_one)

    def test_evidence_one_contributes_zero_risk(self):
        """evidence_support=1 → contribution = 0.0 (no risk from evidence)."""
        features = RiskFeatures(
            uncertainty=0.0, disagreement=0.0, agent_trust=1.0,
            claim_complexity=0.0, evidence_support=1.0,
        )
        score = RiskGate().compute_score(features)
        assert score == pytest.approx(0.0, abs=1e-9)


# =====================================================================
# 15 (extended). Custom weights
# =====================================================================

class TestCustomWeights:

    def test_zero_weight_feature_ignored(self):
        """A feature with weight 0 should not affect the score."""
        config = RiskGateConfig(weights={
            "uncertainty": 0.0, "disagreement": 0.25,
            "agent_trust": 0.25, "claim_complexity": 0.25,
            "evidence_support": 0.25,
        })
        gate = RiskGate(config)

        features_a = RiskFeatures(
            uncertainty=0.0, disagreement=0.5, agent_trust=0.5,
            claim_complexity=0.5, evidence_support=0.5,
        )
        features_b = RiskFeatures(
            uncertainty=1.0, disagreement=0.5, agent_trust=0.5,
            claim_complexity=0.5, evidence_support=0.5,
        )
        assert gate.compute_score(features_a) == gate.compute_score(features_b)

    def test_heavy_uncertainty_weight(self):
        """High weight on uncertainty should amplify its effect."""
        config = RiskGateConfig(weights={
            "uncertainty": 0.80, "disagreement": 0.05,
            "agent_trust": 0.05, "claim_complexity": 0.05,
            "evidence_support": 0.05,
        })
        gate = RiskGate(config)
        low = gate.compute_score(RiskFeatures(
            uncertainty=0.1, disagreement=0.5, agent_trust=0.5,
            claim_complexity=0.5, evidence_support=0.5,
        ))
        high = gate.compute_score(RiskFeatures(
            uncertainty=0.9, disagreement=0.5, agent_trust=0.5,
            claim_complexity=0.5, evidence_support=0.5,
        ))
        # The difference should be large because uncertainty has 80% weight
        assert (high - low) > 0.5


# =====================================================================
# 16 (extended). Custom thresholds
# =====================================================================

class TestCustomThresholds:

    def test_narrow_medium_band(self):
        """thresholds=(0.4, 0.6): narrower MEDIUM band."""
        config = RiskGateConfig(thresholds=(0.4, 0.6))
        gate = RiskGate(config)

        # 0.39 → LOW
        f_low = RiskFeatures(
            uncertainty=0.39, disagreement=0.39,
            agent_trust=0.61, claim_complexity=0.39,
            evidence_support=0.61,
        )
        assert gate.classify(gate.compute_score(f_low)) == RiskLevel.LOW

        # 0.61 → HIGH
        f_high = RiskFeatures(
            uncertainty=0.61, disagreement=0.61,
            agent_trust=0.39, claim_complexity=0.61,
            evidence_support=0.39,
        )
        assert gate.classify(gate.compute_score(f_high)) == RiskLevel.HIGH

    def test_invalid_thresholds_rejected(self):
        with pytest.raises(ValueError, match="Thresholds"):
            RiskGateConfig(thresholds=(0.66, 0.33))  # reversed

    def test_equal_thresholds_rejected(self):
        with pytest.raises(ValueError, match="Thresholds"):
            RiskGateConfig(thresholds=(0.5, 0.5))


# =====================================================================
# 17 (extended). Missing-signal fallback & signals_available
# =====================================================================

class TestMissingSignalFallback:

    def test_all_none_uses_fallback(self):
        """All None → all replaced with unavailable_value (0.5)."""
        gate = RiskGate()
        score = gate.compute_score(RiskFeatures())
        assert score == pytest.approx(0.5, abs=1e-9)

    def test_custom_fallback_value(self):
        """unavailable_value=0.0 → missing signals contribute 0 risk."""
        config = RiskGateConfig(unavailable_value=0.0)
        gate = RiskGate(config)
        # All None → fallback 0.0
        # uncertainty=0, disagreement=0, trust→(1-0)=1, complexity=0, evidence→(1-0)=1
        # score = (0 + 0 + 0.2 + 0 + 0.2) / 1.0 = 0.4
        score = gate.compute_score(RiskFeatures())
        # trust and evidence are inverted: (1 - 0.0) = 1.0
        expected = 0.20 * 0.0 + 0.20 * 0.0 + 0.20 * 1.0 + 0.20 * 0.0 + 0.20 * 1.0
        assert score == pytest.approx(expected, abs=1e-9)

    def test_signals_available_tracks_provided(self):
        features = RiskFeatures(uncertainty=0.5, claim_complexity=0.3)
        result = RiskGate().assess_claim("test", features)
        assert set(result["signals_available"]) == {"uncertainty", "claim_complexity"}
        assert "disagreement" in result["signals_missing"]
        assert "agent_trust" in result["signals_missing"]
        assert "evidence_support" in result["signals_missing"]

    def test_signals_available_all_none(self):
        result = RiskGate().assess_claim("test", RiskFeatures())
        assert result["signals_available"] == []
        assert set(result["signals_missing"]) == set(FEATURE_NAMES)

    def test_signals_available_all_provided(self):
        features = RiskFeatures(
            uncertainty=0.5, disagreement=0.5, agent_trust=0.5,
            claim_complexity=0.5, evidence_support=0.5,
        )
        result = RiskGate().assess_claim("test", features)
        assert set(result["signals_available"]) == set(FEATURE_NAMES)
        assert result["signals_missing"] == []

    def test_none_never_represented_as_observed(self):
        """In the output features dict, None values must stay None,
        not be replaced by the fallback."""
        result = RiskGate().assess_claim("test", RiskFeatures())
        for name in FEATURE_NAMES:
            assert result["features"][name] is None, (
                f"Feature '{name}' should be None in output, not the fallback"
            )


# =====================================================================
# Integration: existing adaptive_agent_router compatibility
# =====================================================================

class TestRouterIntegration:
    """Verify that the Risk Gate output works with the existing router."""

    def test_router_reads_score_key(self):
        """Simulate the router logic on Risk Gate output."""
        from adaptive_routing.core import ALPHA_1, ALPHA_2

        gate = RiskGate()
        result = gate.assess_claims(["claim 1", "claim 2", "claim 3"])
        for cr in result["claim_risks"]:
            score = cr["score"]
            if score < ALPHA_1:
                action = "PASS"
            elif score < ALPHA_2:
                action = "RECHECK"
            else:
                action = "VERIFY"
            # Just verify it doesn't crash and produces valid routing
            assert action in {"PASS", "RECHECK", "VERIFY"}

    def test_graph_compiles(self):
        """The routing graph must still compile after our changes."""
        from adaptive_routing.core import assemble_routing_graph
        graph = assemble_routing_graph()
        assert graph is not None


# =====================================================================
# Unavailable-value config validation
# =====================================================================

class TestUnavailableValueValidation:
    """unavailable_value must be in [0, 1]."""

    def test_negative_unavailable_value_rejected(self):
        with pytest.raises(ValueError, match="unavailable_value"):
            RiskGateConfig(unavailable_value=-0.1)

    def test_above_one_unavailable_value_rejected(self):
        with pytest.raises(ValueError, match="unavailable_value"):
            RiskGateConfig(unavailable_value=1.5)

    def test_boundary_zero_valid(self):
        config = RiskGateConfig(unavailable_value=0.0)
        assert config.unavailable_value == 0.0

    def test_boundary_one_valid(self):
        config = RiskGateConfig(unavailable_value=1.0)
        assert config.unavailable_value == 1.0


# =====================================================================
# risk_level matches score (explicit consistency)
# =====================================================================

class TestRiskLevelMatchesScore:
    """The string risk_level must be consistent with the numeric score."""

    @pytest.mark.parametrize("target_score,expected_level", [
        (0.0,  "LOW"),
        (0.15, "LOW"),
        (0.32, "LOW"),
        (0.33, "MEDIUM"),
        (0.50, "MEDIUM"),
        (0.65, "MEDIUM"),
        (0.66, "HIGH"),
        (0.80, "HIGH"),
        (1.0,  "HIGH"),
    ])
    def test_level_consistent_with_score(self, target_score, expected_level):
        """Build features that produce the target score, then verify
        the returned risk_level matches the expected classification."""
        features = RiskFeatures(
            uncertainty=target_score,
            disagreement=target_score,
            agent_trust=1.0 - target_score,
            claim_complexity=target_score,
            evidence_support=1.0 - target_score,
        )
        result = RiskGate().assess_claim("test", features)
        assert abs(result["risk_score"] - target_score) < 1e-9
        assert result["risk_level"] == expected_level


# =====================================================================
# No LLM / network / API dependency
# =====================================================================

class TestNoExternalDependency:
    """The Risk Gate must not depend on LLM, network, or API services."""

    def test_no_llm_imports_in_gate(self):
        from src.risk_gate import gate as gate_module
        source = inspect.getsource(gate_module)
        for banned in [
            "openai", "langchain", "ChatOpenAI", "requests", "httpx",
            "urllib", "aiohttp", "anthropic",
        ]:
            assert banned not in source, (
                f"Risk Gate source must not reference '{banned}'"
            )

    def test_no_llm_imports_in_features(self):
        from src.risk_gate import features as features_module
        source = inspect.getsource(features_module)
        for banned in [
            "openai", "langchain", "ChatOpenAI", "requests", "httpx",
            "urllib", "aiohttp", "anthropic",
        ]:
            assert banned not in source, (
                f"Features source must not reference '{banned}'"
            )


# =====================================================================
# Risk Gate does not modify input features
# =====================================================================

class TestInputImmutability:
    """The Risk Gate must not mutate the input RiskFeatures."""

    def test_features_unchanged_after_scoring(self):
        features = RiskFeatures(
            uncertainty=0.7,
            disagreement=0.3,
            agent_trust=0.8,
            claim_complexity=0.4,
            evidence_support=0.6,
        )
        # Capture original values
        original = features.to_dict()
        gate = RiskGate()
        gate.assess_claim("test", features)
        # Verify nothing changed
        after = features.to_dict()
        assert original == after

    def test_features_unchanged_after_batch(self):
        f1 = RiskFeatures(uncertainty=0.2)
        f2 = RiskFeatures(uncertainty=0.9, agent_trust=0.1)
        orig1 = f1.to_dict()
        orig2 = f2.to_dict()
        gate = RiskGate()
        gate.assess_claims(["a", "b"], [f1, f2])
        assert f1.to_dict() == orig1
        assert f2.to_dict() == orig2

    def test_frozen_dataclass_prevents_mutation(self):
        """RiskFeatures is frozen — direct attribute assignment must fail."""
        features = RiskFeatures(uncertainty=0.5)
        with pytest.raises(AttributeError):
            features.uncertainty = 0.9  # type: ignore


# =====================================================================
# All-unavailable → 0.5 → MEDIUM (explicit named test)
# =====================================================================

class TestAllUnavailableMediumFallback:
    """When no signals are available, score = 0.5 → MEDIUM."""

    def test_no_features_gives_medium(self):
        result = RiskGate().assess_claim("test")
        assert result["risk_score"] == pytest.approx(0.5, abs=1e-9)
        assert result["risk_level"] == "MEDIUM"

    def test_explicit_all_none_gives_medium(self):
        result = RiskGate().assess_claim("test", RiskFeatures())
        assert result["risk_score"] == pytest.approx(0.5, abs=1e-9)
        assert result["risk_level"] == "MEDIUM"


# =====================================================================
# BLOCKED_FIELDS covers benchmark_schema.EVAL_ONLY_FIELDS
# =====================================================================

class TestBlockedFieldsCoverage:
    """BLOCKED_FIELDS must be a superset of benchmark EVAL_ONLY_FIELDS."""

    def test_blocked_fields_cover_eval_only_fields(self):
        from src.data.benchmark_schema import EVAL_ONLY_FIELDS
        for field in EVAL_ONLY_FIELDS:
            assert field in BLOCKED_FIELDS, (
                f"benchmark_schema.EVAL_ONLY_FIELDS contains '{field}' "
                f"which is missing from risk_gate.BLOCKED_FIELDS"
            )

