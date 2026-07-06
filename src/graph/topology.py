import os
import time
from typing import Dict, Any, Literal, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.metrics.entropy import compute_sequence_entropy
from src.metrics.semantic import calculate_semantic_drift
from src.controllers.atg_matrix import update_adaptive_trust_graph

# Load environment configurations
load_dotenv()

def get_vllm_client() -> ChatOpenAI:
    vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "casperhansen/llama-3-8b-instruct-awq")
    return ChatOpenAI(
        model=model_name,
        openai_api_key="local",
        openai_api_base=vllm_url,
        temperature=0.7,
        model_kwargs={"logprobs": True, "top_logprobs": 5}
    )

class SwarmNodes:
    """
    Implements Swarm Nodes representing Generator, Critic, Judge, and Recovery agents.
    """
    def __init__(self):
        self.client = get_vllm_client()
        
    def generator_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        current_turn = state.get("current_turn", 1)
        history = state.get("history", [])
        
        sys_instr = (
            "You are the Generator agent. Solve the query using step-by-step logic. "
            "At the end of your response, always state your confidence: 'Confidence: XX%'."
        )
        
        messages = [SystemMessage(content=sys_instr)]
        for h in history:
            role_label = "Generator previous" if h["role"] == "generator" else "Critic review"
            messages.append(HumanMessage(content=f"{role_label}: {h['text']}"))
            
        messages.append(HumanMessage(content=f"Query: {query}"))
        
        start_time = time.perf_counter()
        response = self.client.invoke(messages)
        latency = time.perf_counter() - start_time
        
        response_text = response.content
        entropy = compute_sequence_entropy(response.response_metadata, response_text)
        drift_data = calculate_semantic_drift(response_text, query)
        
        metric_payload = {
            "turn": current_turn,
            "node": "generator",
            "entropy": float(entropy),
            "drift": drift_data["drift"],
            "latency": float(latency),
            "inference_time": float(latency),
            "metric_calculation_time": drift_data["latency"]
        }
        
        new_hist = list(history)
        new_hist.append({"role": "generator", "text": response_text})
        
        new_metrics = list(state.get("metrics", []))
        new_metrics.append(metric_payload)
        
        return {
            "history": new_hist,
            "metrics": new_metrics,
            "current_turn": current_turn + 1
        }
        
    def critic_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        current_turn = state.get("current_turn", 1)
        history = state.get("history", [])
        
        latest_gen = ""
        for h in reversed(history):
            if h["role"] == "generator":
                latest_gen = h["text"]
                break
                
        sys_instr = (
            "You are the Critic agent. Critique the Generator's solution. "
            "Identify flaws or confirm calculations. State your confidence: 'Confidence: XX%'."
        )
        
        messages = [
            SystemMessage(content=sys_instr),
            HumanMessage(content=f"Original Query: {query}"),
            HumanMessage(content=f"Generator Solution: {latest_gen}")
        ]
        
        start_time = time.perf_counter()
        response = self.client.invoke(messages)
        latency = time.perf_counter() - start_time
        
        response_text = response.content
        entropy = compute_sequence_entropy(response.response_metadata, response_text)
        drift_data = calculate_semantic_drift(response_text, latest_gen)
        
        metric_payload = {
            "turn": current_turn,
            "node": "critic",
            "entropy": float(entropy),
            "drift": drift_data["drift"],
            "latency": float(latency),
            "inference_time": float(latency),
            "metric_calculation_time": drift_data["latency"]
        }
        
        new_hist = list(history)
        new_hist.append({"role": "critic", "text": response_text})
        
        new_metrics = list(state.get("metrics", []))
        new_metrics.append(metric_payload)
        
        return {
            "history": new_hist,
            "metrics": new_metrics,
            "current_turn": current_turn + 1
        }
        
    def judge_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        current_turn = state.get("current_turn", 1)
        history = state.get("history", [])
        
        sys_instr = (
            "You are the Judge agent. Synthesize the Generator and Critic logic. "
            "Resolve issues and output the final validated answer."
        )
        
        messages = [SystemMessage(content=sys_instr)]
        for h in history:
            messages.append(HumanMessage(content=f"{h['role'].capitalize()} output: {h['text']}"))
            
        messages.append(HumanMessage(content=f"Finalize consensus to: {query}"))
        
        start_time = time.perf_counter()
        response = self.client.invoke(messages)
        latency = time.perf_counter() - start_time
        
        response_text = response.content
        entropy = compute_sequence_entropy(response.response_metadata, response_text)
        drift_data = calculate_semantic_drift(response_text, query)
        
        metric_payload = {
            "turn": current_turn,
            "node": "judge",
            "entropy": float(entropy),
            "drift": drift_data["drift"],
            "latency": float(latency),
            "inference_time": float(latency),
            "metric_calculation_time": drift_data["latency"]
        }
        
        new_hist = list(history)
        new_hist.append({"role": "judge", "text": response_text})
        
        new_metrics = list(state.get("metrics", []))
        new_metrics.append(metric_payload)
        
        return {
            "history": new_hist,
            "metrics": new_metrics,
            "current_turn": current_turn + 1
        }

    def recovery_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Specialized isolation recovery node.
        Strips contaminated context, resets prompt sequence, and forces fresh generation.
        """
        query = state.get("query", "")
        ground_truth = state.get("ground_truth", "")
        current_turn = state.get("current_turn", 1)
        
        print(f"[RECOVERY NODE] Quarantining corrupted context history of size {len(state.get('history', []))}.")
        
        # Hard reset: clean history records and keep only clean query anchor info
        clean_history = [
            {"role": "system", "text": "Corrupted history quarantined. Rerunning reasoning loop from clean context."}
        ]
        
        sys_instr = (
            "You are the Recovery agent. The preceding generation path was quarantined due to high entropy/drift. "
            "Resolve the problem from scratch. Prioritize absolute accuracy. Write the final answer at the end."
        )
        
        messages = [
            SystemMessage(content=sys_instr),
            HumanMessage(content=f"Query: {query}"),
            HumanMessage(content=f"Original Ground Truth Hint: {ground_truth}" if ground_truth else f"Query: {query}")
        ]
        
        start_time = time.perf_counter()
        response = self.client.invoke(messages)
        latency = time.perf_counter() - start_time
        
        response_text = response.content
        entropy = compute_sequence_entropy(response.response_metadata, response_text)
        drift_data = calculate_semantic_drift(response_text, query)
        
        metric_payload = {
            "turn": current_turn,
            "node": "recovery",
            "entropy": float(entropy),
            "drift": drift_data["drift"],
            "latency": float(latency),
            "inference_time": float(latency),
            "metric_calculation_time": drift_data["latency"]
        }
        
        clean_history.append({"role": "judge", "text": response_text})
        
        new_metrics = list(state.get("metrics", []))
        new_metrics.append(metric_payload)
        
        # Restore trust scores to baseline to exit recovery cleanly
        restored_scores = {
            "generator->critic": 1.0,
            "critic->judge": 1.0
        }
        
        return {
            "history": clean_history,
            "metrics": new_metrics,
            "current_turn": current_turn + 1,
            "trust_scores": restored_scores
        }

def execution_router(state: AgentState) -> Literal["generator", "critic", "judge", "fallback_recovery", "__end__"]:
    """
    State router. Reroutes to 'fallback_recovery' instead of ending immediately 
    to preserve generation capabilities when trust bounds are breached.
    """
    trust_scores = state.get("trust_scores", {})
    config = state.get("configuration", {})
    isolation_threshold = config.get("isolation_threshold", 0.25)
    
    for edge, val in trust_scores.items():
        if val < isolation_threshold:
            print(f"[ATG ROUTER] Dynamic recovery triggered. Edge '{edge}' weight {val:.4f} < {isolation_threshold}.")
            return "fallback_recovery"
            
    history = state.get("history", [])
    current_turn = state.get("current_turn", 1)
    max_turns = state.get("max_turns", 6)
    
    if current_turn >= max_turns:
        return END
        
    if not history:
        return "generator"
        
    last_role = history[-1]["role"]
    if last_role == "generator":
        return "critic"
    elif last_role == "critic":
        if current_turn < max_turns - 1:
            return "generator"
        return "judge"
    elif last_role == "judge":
        return END
        
    return END

# ATG weight update controller nodes
def controller_gen_critic_node(state: AgentState) -> Dict[str, Any]:
    config = state.get("configuration", {})
    alpha = config.get("alpha", 0.5)
    beta = config.get("beta", 0.15)
    updated_scores = update_adaptive_trust_graph(state, "generator->critic", alpha, beta)
    return {"trust_scores": updated_scores}

def controller_critic_judge_node(state: AgentState) -> Dict[str, Any]:
    config = state.get("configuration", {})
    alpha = config.get("alpha", 0.5)
    beta = config.get("beta", 0.15)
    updated_scores = update_adaptive_trust_graph(state, "critic->judge", alpha, beta)
    return {"trust_scores": updated_scores}

def assemble_atg_graph(nodes_instance: SwarmNodes) -> StateGraph:
    """
    Assembles and compiles the StateGraph workflow with Recovery mechanics.
    """
    workflow = StateGraph(AgentState)
    
    # Add Swarm nodes
    workflow.add_node("generator", nodes_instance.generator_node)
    workflow.add_node("critic", nodes_instance.critic_node)
    workflow.add_node("judge", nodes_instance.judge_node)
    workflow.add_node("recovery", nodes_instance.recovery_node)
    
    # Add controllers
    workflow.add_node("controller_gen_critic", controller_gen_critic_node)
    workflow.add_node("controller_critic_judge", controller_critic_judge_node)
    
    workflow.set_entry_point("generator")
    
    workflow.add_edge("generator", "controller_gen_critic")
    workflow.add_conditional_edges(
        "controller_gen_critic",
        execution_router,
        {
            "generator": "generator",
            "critic": "critic",
            "judge": "judge",
            "fallback_recovery": "recovery",
            "__end__": END
        }
    )
    
    workflow.add_edge("critic", "controller_critic_judge")
    workflow.add_conditional_edges(
        "controller_critic_judge",
        execution_router,
        {
            "generator": "generator",
            "critic": "critic",
            "judge": "judge",
            "fallback_recovery": "recovery",
            "__end__": END
        }
    )
    
    workflow.add_edge("recovery", END)
    workflow.add_edge("judge", END)
    
    return workflow.compile()