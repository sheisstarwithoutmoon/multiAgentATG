import math

def update_adaptive_trust_graph(state: dict, source_edge: str, alpha: float = 0.5) -> dict:
    """
    Inspects dynamic metric values against your calibrated baseline thresholds.
    Damps the graph adjacency matrix weights exponentially during structural anomalies.
    """
    updated_scores = state["trust_scores"].copy()
    latest_metric = state["metrics"][-1]
    
    entropy = latest_metric["entropy"]
    drift = latest_metric["drift"]
    
    tau = state["calibration_bounds"].get("entropy_threshold", 0.7)
    theta = state["calibration_bounds"].get("drift_threshold", 0.4)
    
    # Check for threshold violations
    if entropy > tau or drift > theta:
        current_weight = updated_scores.get(source_edge, 1.0)
        # Exponential attenuation equation
        updated_scores[source_edge] = current_weight * alpha
        
    return updated_scores