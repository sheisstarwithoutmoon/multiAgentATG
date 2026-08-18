"""
adaptive_routing/metrics/compute.py
All metric computation functions for the adaptive routing step.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


# Agent calls per routing path
AGENT_CALLS = {
    "PASS":    1,   # Agent 1 only
    "RECHECK": 2,   # Agent 1 + Agent 2
    "VERIFY":  4,   # Agent 1 + Agent 2 + Agent 3 + Agent 4
}
FIXED_BASELINE_CALLS = 4   # Always-use-all-4-agents baseline


@dataclass
class RoutingRecord:
    """One claim's routing decision + ground truth label."""
    claim:            str
    risk_score:       float
    decision:         str    # "PASS" | "RECHECK" | "VERIFY"
    is_hallucination: bool   # Ground truth: was this claim actually wrong?
    escalated:        bool = False  # Was RECHECK escalated to VERIFY via conflict?


@dataclass
class RouterMetricsResult:
    """All computed metrics for one evaluation run."""
    n_claims:            int
    distribution:        Dict[str, float]
    precision:           float
    recall:              float
    f1:                  float
    avg_agents_adaptive: float
    avg_agents_baseline: float
    agent_savings_pct:   float
    escalation_rate:     float
    alpha_1:             float
    alpha_2:             float


def compute_routing_distribution(records: List[RoutingRecord]) -> Dict[str, float]:
    """% of claims routed to each path (PASS / RECHECK / VERIFY)."""
    n = len(records)
    if n == 0:
        return {"PASS": 0.0, "RECHECK": 0.0, "VERIFY": 0.0}
    counts = {"PASS": 0, "RECHECK": 0, "VERIFY": 0}
    for r in records:
        counts[r.decision] = counts.get(r.decision, 0) + 1
    return {k: round(v / n * 100, 1) for k, v in counts.items()}


def compute_routing_accuracy(records: List[RoutingRecord]) -> Tuple[float, float, float]:
    """
    RECHECK or VERIFY = flagged (positive class).
    Hallucinated claims should be flagged; correct claims should PASS.
    Returns (precision, recall, f1).
    """
    tp = fp = fn = tn = 0
    for r in records:
        flagged = r.decision in ("RECHECK", "VERIFY")
        if flagged and r.is_hallucination:
            tp += 1
        elif flagged and not r.is_hallucination:
            fp += 1
        elif not flagged and r.is_hallucination:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_agent_savings(records: List[RoutingRecord]) -> Tuple[float, float, float]:
    """
    Avg agent calls per claim: adaptive routing vs fixed 4-agent baseline.
    RECHECK claims that were escalated count as 4 (all agents ran).
    Returns (avg_adaptive_calls, avg_baseline_calls, savings_pct).
    """
    if not records:
        return 0.0, float(FIXED_BASELINE_CALLS), 0.0

    total_adaptive = 0
    for r in records:
        if r.decision == "RECHECK" and r.escalated:
            total_adaptive += 4   # Escalated: Agent 3 + 4 also ran
        else:
            total_adaptive += AGENT_CALLS.get(r.decision, 4)

    avg_adaptive = total_adaptive / len(records)
    savings_pct  = (1 - avg_adaptive / FIXED_BASELINE_CALLS) * 100
    return round(avg_adaptive, 3), float(FIXED_BASELINE_CALLS), round(savings_pct, 1)


def compute_escalation_rate(records: List[RoutingRecord]) -> float:
    """% of RECHECK claims further escalated to VERIFY by conflict_router."""
    recheck = [r for r in records if r.decision == "RECHECK"]
    if not recheck:
        return 0.0
    escalated = sum(1 for r in recheck if r.escalated)
    return round(escalated / len(recheck) * 100, 1)


def evaluate(
    records: List[RoutingRecord],
    alpha_1: float = 0.33,
    alpha_2: float = 0.66,
) -> RouterMetricsResult:
    """Run all router metrics on a list of RoutingRecord entries."""
    distribution = compute_routing_distribution(records)
    precision, recall, f1 = compute_routing_accuracy(records)
    avg_adaptive, avg_baseline, savings_pct = compute_agent_savings(records)
    escalation_rate = compute_escalation_rate(records)

    return RouterMetricsResult(
        n_claims=len(records),
        distribution=distribution,
        precision=precision,
        recall=recall,
        f1=f1,
        avg_agents_adaptive=avg_adaptive,
        avg_agents_baseline=avg_baseline,
        agent_savings_pct=savings_pct,
        escalation_rate=escalation_rate,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
    )


def threshold_sensitivity(
    records_fn,
    alpha_grid: List[Tuple[float, float]] = None,
) -> List[Dict]:
    """
    Sweep over (α1, α2) pairs and report how metrics change.
    records_fn(alpha_1, alpha_2) -> List[RoutingRecord]
    """
    if alpha_grid is None:
        alpha_grid = [
            (0.20, 0.50),
            (0.25, 0.55),
            (0.33, 0.66),
            (0.40, 0.70),
            (0.45, 0.75),
        ]
    results = []
    for a1, a2 in alpha_grid:
        records = records_fn(a1, a2)
        m = evaluate(records, a1, a2)
        results.append({
            "alpha_1":   a1,
            "alpha_2":   a2,
            "precision": m.precision,
            "recall":    m.recall,
            "f1":        m.f1,
            "savings_%": m.agent_savings_pct,
        })
    return results


def print_report(result: RouterMetricsResult) -> None:
    """Print a clean summary of all router metrics."""
    print("\n" + "=" * 54)
    print(f"  ADAPTIVE ROUTER — EVALUATION REPORT")
    print(f"  α1={result.alpha_1}  α2={result.alpha_2}  n={result.n_claims} claims")
    print("=" * 54)

    print("\n[1] Routing Distribution")
    for path, pct in result.distribution.items():
        bar = "#" * int(pct / 5)
        print(f"    {path:8s}  {pct:5.1f}%  {bar}")

    print("\n[2] Routing Accuracy (vs ground truth hallucination labels)")
    print(f"    Precision : {result.precision:.4f}")
    print(f"    Recall    : {result.recall:.4f}")
    print(f"    F1        : {result.f1:.4f}")

    print("\n[3] Agent Call Efficiency")
    print(f"    Avg calls — Adaptive : {result.avg_agents_adaptive:.2f}")
    print(f"    Avg calls — Baseline : {result.avg_agents_baseline:.2f}  (fixed 4-agent)")
    print(f"    LLM Call Savings     : {result.agent_savings_pct:.1f}%")

    print("\n[4] Escalation Rate (RECHECK → VERIFY via conflict)")
    print(f"    {result.escalation_rate:.1f}% of RECHECK claims escalated further")

    print("=" * 54)
