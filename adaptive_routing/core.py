"""
adaptive_routing/core.py
Adaptive routing layer for the AHC pipeline.
Owns: RoutingState, adaptive_agent_router, conflict_router, graph topology.
Risk gate, agent nodes, and verification are implemented by teammates.
"""

import os
import random
from typing import Dict, Any, List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from src.risk_gate.features import RiskFeatures
from src.risk_gate.gate import RiskGate
from adaptive_routing.agents import (
    agent_1_generate_and_extract,
    agent_1_compute_signals,
    agent_2_recheck,
    agent_3_compare,
    agent_3_verify,
    agent_4_adjudicate
)


import yaml

# Load thresholds from config
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "ahc_config.yaml")
try:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        ALPHA_1 = float(cfg.get("risk_gate", {}).get("alpha_1", 0.33))
        ALPHA_2 = float(cfg.get("risk_gate", {}).get("alpha_2", 0.66))
except Exception:
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

def run_generator(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Agent 1 generating and extracting claims...")
    query = state.get("query", "")
    answer, claims = agent_1_generate_and_extract(query)
    
    # Store the initial answer in the state, even if it might be changed later
    return {
        "final_answer": answer, 
        "claims": claims
    }

def run_risk_gate(state: RoutingState) -> Dict[str, Any]:
    """Risk Gate: computes signals and routes."""
    print("[Node] Risk Gate calculating scores...")
    query = state.get("query", "")
    claims = state.get("claims", [])
    initial_answer = state.get("final_answer", "")
    
    # Compute signals for each claim
    features_dicts = agent_1_compute_signals(query, initial_answer, claims)
    
    # Convert to RiskFeatures objects
    features_list = [RiskFeatures(**fd) for fd in features_dicts]
    
    gate = RiskGate()
    return gate.assess_claims(claims, features_list)

def run_agent_2(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Agent 2 processing medium-risk claims...")
    query = state.get("query", "")
    claim_risks = state.get("claim_risks", [])
    
    medium_risk_claims = [cr["claim"] for cr in claim_risks if ALPHA_1 <= cr["score"] < ALPHA_2]
    
    answer_2 = agent_2_recheck(query, medium_risk_claims)
    
    # We will temporarily store Agent 2's answer in the state for the Compare node
    return {
        "agent_2_assessments": [{"answer": answer_2}] # Store in a list to match TypedDict
    }

def run_compare(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Compare resolving agreement or conflict...")
    query = state.get("query", "")
    answer_1 = state.get("final_answer", "")
    
    agent_2_assessments = state.get("agent_2_assessments", [])
    answer_2 = agent_2_assessments[0]["answer"] if agent_2_assessments else ""
    
    if not answer_2:
        return {"final_status": "AGREEMENT", "has_conflict": False}
        
    is_conflict = agent_3_compare(query, answer_1, answer_2)
    
    if is_conflict:
        return {"final_status": "CONFLICT", "has_conflict": True}
    return {"final_status": "AGREEMENT", "has_conflict": False}

def run_verify(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Verify Claims checking evidence for high-risk claims...")
    query = state.get("query", "")
    claim_risks = state.get("claim_risks", [])
    has_conflict = state.get("has_conflict", False)
    
    # Verify HIGH risk claims, or ALL claims if there was a conflict
    claims_to_verify = []
    if has_conflict:
        claims_to_verify = [cr["claim"] for cr in claim_risks]
    else:
        claims_to_verify = [cr["claim"] for cr in claim_risks if cr["score"] >= ALPHA_2]
        
    verified_results = agent_3_verify(query, claims_to_verify)
    return {"verified_claims": verified_results}

def run_adjudicator(state: RoutingState) -> Dict[str, Any]:
    print("[Node] Adjudicator deciding final answer...")
    query = state.get("query", "")
    answer_1 = state.get("final_answer", "")
    
    agent_2_assessments = state.get("agent_2_assessments", [])
    answer_2 = agent_2_assessments[0]["answer"] if agent_2_assessments else ""
    
    verified_results = state.get("verified_claims", [])
    
    final_answer = agent_4_adjudicate(query, answer_1, answer_2, verified_results)
    
    return {
        "final_answer": final_answer,
        "final_status": "Final answer (from High Risk path)"
    }


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

    g.add_node("generator",      run_generator)
    g.add_node("risk_gate",      run_risk_gate)
    g.add_node("agent_2",        run_agent_2)
    g.add_node("compare",        run_compare)
    g.add_node("verify_claims",  run_verify)
    g.add_node("adjudicator",    run_adjudicator)
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
