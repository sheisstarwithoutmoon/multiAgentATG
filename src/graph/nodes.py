import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.metrics.entropy import compute_sequence_entropy
from src.metrics.semantic import calculate_semantic_drift
from src.graph.state import AgentState

class ModelSwarmNodes:
    """
    Implements Swarm Nodes for multi-agent execution:
    - GeneratorNode: Generates candidate responses/reasoning.
    - CriticNode: Critiques generator outputs, analyzing potential fallacies.
    - JudgeNode: Compiles and aggregates consensus final verification.
    """
    
    def __init__(self, model_name: str = "casperhansen/llama-3-8b-instruct-awq", temperature: float = 0.3):
        # Initialize ChatOpenAI client with logprobs enabled
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            model_kwargs={"logprobs": True, "top_logprobs": 5}
        )
        
    def generator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the generator node. Generates a solution to the query.
        """
        query = state.get("query", "")
        current_turn = state.get("current_turn", 1)
        history = state.get("history", [])
        
        system_instructions = (
            "You are the Generator agent. Your task is to provide a clear, logical, "
            "step-by-step solution to the problem presented. Focus on accuracy and details."
        )
        
        messages = [SystemMessage(content=system_instructions)]
        for h in history:
            if h["role"] == "generator":
                messages.append(HumanMessage(content=f"Your previous response: {h['text']}"))
            elif h["role"] == "critic":
                messages.append(HumanMessage(content=f"Critic feedback: {h['text']}"))
                
        messages.append(HumanMessage(content=f"Solve the query: {query}"))
        
        start_time = time.time()
        response = self.llm.invoke(messages)
        latency = time.time() - start_time
        
        response_text = response.content
        entropy = compute_sequence_entropy(response.response_metadata)
        
        # Calculate semantic drift relative to original query or anchor point (e.g. prompt query)
        drift = calculate_semantic_drift(response_text, query)
        
        # Log telemetry trace
        metric_payload = {
            "turn": current_turn,
            "node": "generator",
            "entropy": float(entropy),
            "drift": float(drift),
            "latency": float(latency)
        }
        
        new_history = list(history)
        new_history.append({"role": "generator", "text": response_text})
        
        new_metrics = list(state.get("metrics", []))
        new_metrics.append(metric_payload)
        
        return {
            "history": new_history,
            "metrics": new_metrics,
            "current_turn": current_turn + 1
        }
        
    def critic_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the critic node. Reviews the generator's latest solution.
        """
        query = state.get("query", "")
        current_turn = state.get("current_turn", 1)
        history = state.get("history", [])
        
        # Find latest generator response in history
        latest_gen_resp = ""
        for h in reversed(history):
            if h["role"] == "generator":
                latest_gen_resp = h["text"]
                break
                
        system_instructions = (
            "You are the Critic agent. Examine the generator's proposed solution carefully. "
            "Identify any calculation errors, semantic inconsistencies, or logical leaps. "
            "Write a critique and point out specific flaws, or confirm if the solution is completely sound."
        )
        
        messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=f"Original Query: {query}"),
            HumanMessage(content=f"Proposed Generator Solution: {latest_gen_resp}")
        ]
        
        start_time = time.time()
        response = self.llm.invoke(messages)
        latency = time.time() - start_time
        
        response_text = response.content
        entropy = compute_sequence_entropy(response.response_metadata)
        
        # Compute drift from anchor context (generator response)
        drift = calculate_semantic_drift(response_text, latest_gen_resp)
        
        metric_payload = {
            "turn": current_turn,
            "node": "critic",
            "entropy": float(entropy),
            "drift": float(drift),
            "latency": float(latency)
        }
        
        new_history = list(history)
        new_history.append({"role": "critic", "text": response_text})
        
        new_metrics = list(state.get("metrics", []))
        new_metrics.append(metric_payload)
        
        return {
            "history": new_history,
            "metrics": new_metrics,
            "current_turn": current_turn + 1
        }
        
    def judge_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the judge node. Reviews history and returns the final verified consensus response.
        """
        query = state.get("query", "")
        current_turn = state.get("current_turn", 1)
        history = state.get("history", [])
        
        system_instructions = (
            "You are the Judge agent. Compile the step-by-step reasoning from the generator "
            "and critic reports. Resolve any conflicts, perform final arithmetic/logical verification, "
            "and output the final definitive answer."
        )
        
        messages = [SystemMessage(content=system_instructions)]
        for h in history:
            messages.append(HumanMessage(content=f"{h['role'].capitalize()} output: {h['text']}"))
            
        messages.append(HumanMessage(content=f"Synthesize final verified consensus answer to query: {query}"))
        
        start_time = time.time()
        response = self.llm.invoke(messages)
        latency = time.time() - start_time
        
        response_text = response.content
        entropy = compute_sequence_entropy(response.response_metadata)
        
        # Compute semantic drift compared to query grounding
        drift = calculate_semantic_drift(response_text, query)
        
        metric_payload = {
            "turn": current_turn,
            "node": "judge",
            "entropy": float(entropy),
            "drift": float(drift),
            "latency": float(latency)
        }
        
        new_history = list(history)
        new_history.append({"role": "judge", "text": response_text})
        
        new_metrics = list(state.get("metrics", []))
        new_metrics.append(metric_payload)
        
        return {
            "history": new_history,
            "metrics": new_metrics,
            "current_turn": current_turn + 1
        }
