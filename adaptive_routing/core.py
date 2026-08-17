"""
adaptive_routing/core.py
Adaptive routing layer for the AHC pipeline.
Owns: RoutingState, adaptive_agent_router, conflict_router, graph topology.
Risk gate, agent nodes, and verification are implemented by teammates.
"""

import random
from typing import Dict, Any, List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END


ALPHA_1: float = 0.33
ALPHA_2: float = 0.66


class RoutingState(TypedDict):
    query: str
    claims: List[str]
    claim_risks: List[Dict[str, Any]]
    passed_claims: List[str]
    medium_risk_claims: List[str]
    quarantined_claims: List[str]
    agent_2_assessments: List[Dict[str, Any]]
    conflict_flags: List[Dict[str, Any]]
    has_conflict: bool
    verified_claims: List[Dict[str, Any]]
    final_answer: str
    final_status: str


# Stub nodes — teammates replace these with real implementations

def _stub_generator(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Agent 1 generating and extracting claims...")
    return {}

def _stub_risk_gate(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Risk Gate calculating scores...")
    claims = state.get("claims", [])
    risks = []
    for claim in claims:
        score = random.uniform(0, 1.0)
        risks.append({"claim": claim, "score": score})
    return {"claim_risks": risks}

def _stub_agent_2(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Agent 2 processing medium-risk claims...")
    return {}

def _stub_compare(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Compare resolving agreement or conflict...")
    is_conflict = random.choice([True, False])
    if is_conflict:
        return {"final_status": "CONFLICT", "has_conflict": True}
    return {"final_status": "AGREEMENT", "has_conflict": False}

def _stub_verify(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Verify Claims checking evidence for high-risk claims...")
    return {}

def _stub_adjudicator(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Adjudicator deciding final answer...")
    return {"final_status": "Final answer (from High Risk path)"}


def adaptive_agent_router(state: RoutingState) -> Literal["verify_claims", "agent_2", "__end__"]:
    """
    Primary conditional edge after risk_gate.
    Routes based on highest-severity claim bucket.
    Priority: VERIFY > RECHECK > PASS
    """
    print("[Router] Inspecting Risk Scores and deciding routing path...")

    claim_risks = state.get("claim_risks", [])
    has_high_risk = False
    has_medium_risk = False

    for cr in claim_risks:
        score = cr["score"]
        if score < ALPHA_1:
            action = "PASS (Low Risk)"
        elif score < ALPHA_2:
            action = "RECHECK (Medium Risk)"
            has_medium_risk = True
        else:
            action = "VERIFY (High Risk)"
            has_high_risk = True
        print(f"  -> Claim: '{cr['claim']}' | Score: {score:.2f} | Routing Action: {action}")

    if has_high_risk:
        print("[Router Decision] -> Routing to Verify Claims (High Risk Path)")
        return "verify_claims"
    if has_medium_risk:
        print("[Router Decision] -> Routing to Agent 2 (Medium Risk Path)")
        return "agent_2"

    print("[Router Decision] -> All safe. Returning answer directly.")
    return END


def conflict_router(state: RoutingState) -> Literal["verify_claims", "finalise_agree"]:
    """
    Secondary conditional edge after compare.
    Conflict → verify_claims. Agreement → finalise_agree.
    """
    status = state.get("final_status", "")
    print(f"[Conflict Router] Inspecting Compare result: {status}")

    if state.get("has_conflict", False):
        print("[Conflict Router Decision] -> Conflict detected! Rerouting to Verify Claims (Red Path)")
        return "verify_claims"

    print("[Conflict Router Decision] -> Agreement reached. Returning answer directly.")
    return "finalise_agree"


def _finalise_pass(state: RoutingState) -> Dict[str, Any]:
    return {"final_status": "Pending"}

def _finalise_agree(state: RoutingState) -> Dict[str, Any]:
    return {}


def assemble_routing_graph() -> StateGraph:
    """
    Full AHC LangGraph topology.

        generator → risk_gate → [adaptive_agent_router]
            ├─ PASS:    finalise_pass  → END
            ├─ RECHECK: agent_2 → compare → [conflict_router]
            │               ├─ AGREE:    finalise_agree → END
            │               └─ CONFLICT: verify_claims → adjudicator → END
            └─ VERIFY:  verify_claims → adjudicator → END
    """
    g = StateGraph(RoutingState)

    g.add_node("generator",      _stub_generator)
    g.add_node("risk_gate",      _stub_risk_gate)
    g.add_node("agent_2",        _stub_agent_2)
    g.add_node("compare",        _stub_compare)
    g.add_node("verify_claims",  _stub_verify)
    g.add_node("adjudicator",    _stub_adjudicator)
    g.add_node("finalise_pass",  _finalise_pass)
    g.add_node("finalise_agree", _finalise_agree)

    g.set_entry_point("generator")

    g.add_edge("generator",      "risk_gate")
    g.add_edge("agent_2",        "compare")
    g.add_edge("verify_claims",  "adjudicator")
    g.add_edge("adjudicator",    END)
    g.add_edge("finalise_pass",  END)
    g.add_edge("finalise_agree", END)

    g.add_conditional_edges(
        "risk_gate",
        adaptive_agent_router,
        {
            "verify_claims": "verify_claims",
            "agent_2":       "agent_2",
            END:             "finalise_pass",
        },
    )

    g.add_conditional_edges(
        "compare",
        conflict_router,
        {
            "verify_claims":  "verify_claims",
            "finalise_agree": "finalise_agree",
        },
    )

    return g.compile()
