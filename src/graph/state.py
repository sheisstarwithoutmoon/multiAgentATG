import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

# Initialize python-dotenv configuration
load_dotenv()

class AgentState(TypedDict):
    """
    TypedDict representing the unified graph state for the ATG topology controller.
    
    Attributes:
        query: The input prompt/problem statement (potentially mutated).
        ground_truth: The gold-standard response for validation.
        current_turn: Integer tracker for the current iteration loop.
        max_turns: Upper bound turn limit.
        history: Structured dialogue log containing role-based interaction text.
        metrics: List of telemetry dictionaries tracking node performance metrics.
        trust_scores: Active edge weight mappings (e.g., 'generator->critic').
        calibration_bounds: Dynamically determined statistical thresholds for entropy/drift.
        configuration: Evaluator hyperparameter thresholds and settings.
    """
    query: str
    ground_truth: str
    current_turn: int
    max_turns: int
    history: List[Dict[str, str]]
    metrics: List[Dict[str, Any]]
    trust_scores: Dict[str, float]
    calibration_bounds: Dict[str, float]
    configuration: Dict[str, Any]