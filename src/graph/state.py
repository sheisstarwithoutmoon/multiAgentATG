from typing import Dict, List, TypedDict

class AgentState(TypedDict):
    query: str                       # Original dataset prompt question
    ground_truth: str                # Gold standard truth evaluation string
    current_turn: int                # Running tracking indicator for loops
    max_turns: int                   # Bound iteration boundary ceiling
    history: List[Dict[str, str]]    # Shared dialog trail context window
    metrics: List[Dict]              # High-precision performance logs
    trust_scores: Dict[str, float]   # Graph adjacency mapping scores
    calibration_bounds: Dict[str, float] # Dynamically calculated threshold limits