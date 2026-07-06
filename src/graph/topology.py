# src/graph/topology.py
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState

def execution_router(state: AgentState) -> str:
    """
    Inspects active adjacency matrices at runtime. Short-circuits the pipeline 
    instantly if path validation metrics drop below safety limits.
    """
    isolation_threshold = state["calibration_bounds"].get("isolation_threshold", 0.25)
    
    for edge_name, weight in state["trust_scores"].items():
        if weight < isolation_threshold:
            print(f"⚠️ Structural Cutoff Activated: Edge '{edge_name}' dropped to {weight:.4f}. Isolating cascade path.")
            return "end"
            
    if state["current_turn"] >= state["max_turns"]:
        return "end"
        
    return "continue"

def build_cascade_graph(node_instance_class) -> StateGraph:
    """Compiles the orchestration blueprint using local execution logic nodes."""
    workflow = StateGraph(AgentState)
    
    # Register core operational workflow blocks
    workflow.add_node("GeneratorNode", node_instance_class.generator_node)
    workflow.add_node("CriticNode", node_instance_class.critic_node)
    
    workflow.set_entry_point("GeneratorNode")
    workflow.add_edge("GeneratorNode", "CriticNode")
    
    # Wire conditional boundaries to enforce quarantine zones
    workflow.add_conditional_edges(
        "CriticNode",
        execution_router,
        {
            "continue": "GeneratorNode",
            "end": END
        }
    )
    
    return workflow.compile()