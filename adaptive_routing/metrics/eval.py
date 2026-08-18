"""
adaptive_routing/metrics/eval.py
Demo runner — shows metrics output for all 4 routing scenarios.
Uses mock data. Replace with real pipeline records when teammates' code is ready.

Run: python3 eval.py                           (from adaptive_routing/metrics/)
  or: python3 -m adaptive_routing.metrics.eval  (from project root)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adaptive_routing.metrics import (
    RoutingRecord,
    evaluate,
    threshold_sensitivity,
    print_report,
)

ALPHA_1 = 0.33
ALPHA_2 = 0.66


def make_records(score: float, escalated: bool = False) -> list:
    """3 claims with a fixed score. Ground truth: 'sky is green' is a hallucination."""
    claims         = ["The earth is round.", "The sky is green.", "Water is wet."]
    hallucinations = [False, True, False]

    if score < ALPHA_1:
        decision = "PASS"
    elif score < ALPHA_2:
        decision = "RECHECK"
    else:
        decision = "VERIFY"

    return [
        RoutingRecord(
            claim=c,
            risk_score=score,
            decision=decision,
            is_hallucination=h,
            escalated=escalated if decision == "RECHECK" else False,
        )
        for c, h in zip(claims, hallucinations)
    ]


def main():
    print("\n" + "=" * 54)
    print("  ADAPTIVE ROUTER — ALL 4 SCENARIO EVALUATIONS")
    print("=" * 54)

    print("\n>>> SCENARIO 1: ALL LOW RISK (Yellow Path)  score=0.10")
    print_report(evaluate(make_records(0.10), ALPHA_1, ALPHA_2))
    print("  NOTE: Recall=0 — hallucinated claim slipped through PASS.")

    print("\n>>> SCENARIO 2: MEDIUM RISK + AGREEMENT (Orange Path)  score=0.50")
    print_report(evaluate(make_records(0.50, escalated=False), ALPHA_1, ALPHA_2))
    print("  NOTE: RECHECK, Agent 2 agreed. Escalation=0%. 50% savings.")

    print("\n>>> SCENARIO 3: MEDIUM RISK + CONFLICT (Orange→Red)  score=0.50")
    print_report(evaluate(make_records(0.50, escalated=True), ALPHA_1, ALPHA_2))
    print("  NOTE: All RECHECK escalated to VERIFY. Escalation=100%. 0% savings.")

    print("\n>>> SCENARIO 4: HIGH RISK (Red Path)  score=0.90")
    print_report(evaluate(make_records(0.90), ALPHA_1, ALPHA_2))
    print("  NOTE: Direct VERIFY. 0% savings but max recall.")

    print("\n" + "=" * 54)
    print("  THRESHOLD SENSITIVITY: α1, α2 vs Precision/Recall/Savings")
    print("=" * 54)

    def records_for_threshold(a1, a2):
        score = 0.50
        claims = ["The earth is round.", "The sky is green.", "Water is wet."]
        hallu  = [False, True, False]
        decision = "PASS" if score < a1 else ("RECHECK" if score < a2 else "VERIFY")
        return [
            RoutingRecord(claim=c, risk_score=score, decision=decision,
                          is_hallucination=h, escalated=False)
            for c, h in zip(claims, hallu)
        ]

    sweep = threshold_sensitivity(records_for_threshold)
    print(f"\n  {'α1':>6} {'α2':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Savings%':>10}")
    print("  " + "-" * 52)
    for row in sweep:
        print(
            f"  {row['alpha_1']:>6.2f} {row['alpha_2']:>6.2f} "
            f"{row['precision']:>10.4f} {row['recall']:>8.4f} "
            f"{row['f1']:>8.4f} {row['savings_%']:>9.1f}%"
        )
    print("\n  Lower α1 → more PASS → higher savings, lower recall")
    print("  Higher α2 → more RECHECK instead of VERIFY → cheaper but less thorough")


if __name__ == "__main__":
    main()
