import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adaptive_routing.core import assemble_routing_graph, RoutingState

def run_scenario(graph, name: str, mock_score: float, mock_conflict: bool):
    print(f"\n{'='*50}")
    print(f" SCENARIO: {name}")
    print(f"{'='*50}")

    original_uniform = random.uniform
    original_choice = random.choice

    random.uniform = lambda a, b: mock_score
    random.choice = lambda seq: mock_conflict

    initial_state = RoutingState(
        query="Test query",
        claims=["The earth is round.", "The sky is green.", "Water is wet."],
        claim_risks=[],
        passed_claims=[],
        medium_risk_claims=[],
        quarantined_claims=[],
        agent_2_assessments=[],
        conflict_flags=[],
        has_conflict=False,
        verified_claims=[],
        final_answer="",
        final_status="Pending",
    )

    final_state = graph.invoke(initial_state)

    print("\n[RESULT]")
    print(f"Final Status: {final_state.get('final_status', 'Unknown')}")

    random.uniform = original_uniform
    random.choice = original_choice


def main():
    print("Initializing LangGraph Topology...")
    graph = assemble_routing_graph()

    # 1. All Low Risk (PASS path) — score < 0.33
    run_scenario(graph, "TEST 1: ALL LOW RISK (Yellow Path)", mock_score=0.10, mock_conflict=False)

    # 2. Medium Risk + Agreement (RECHECK path) — 0.33 <= score < 0.66, no conflict
    run_scenario(graph, "TEST 2: MEDIUM RISK + AGREEMENT (Orange Path)", mock_score=0.50, mock_conflict=False)

    # 3. Medium Risk + Conflict → Verify (RECHECK → RED path)
    run_scenario(graph, "TEST 3: MEDIUM RISK + CONFLICT (Orange to Red Path)", mock_score=0.50, mock_conflict=True)

    # 4. High Risk (VERIFY path directly) — score >= 0.66
    run_scenario(graph, "TEST 4: HIGH RISK (Red Path)", mock_score=0.90, mock_conflict=False)


if __name__ == "__main__":
    main()
