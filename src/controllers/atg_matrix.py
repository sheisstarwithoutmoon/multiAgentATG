from typing import Dict, Any

def update_adaptive_trust_graph(
    state: Dict[str, Any],
    source_edge: str,
    alpha: float = 0.85,
    beta: float = 0.10
) -> Dict[str, float]:
    """
    Updates the Adaptive Trust Graph weights with dynamic decay and recovery.
    
    Uses thresholds dynamically calculated during the warm-up calibration phase
    and stored in state["calibration_bounds"].
    
    Args:
        state: LangGraph AgentState.
        source_edge: Edge key (e.g., 'generator->critic').
        alpha: Exponential decay factor.
        beta: Linear recovery factor.
        
    Returns:
        Dict[str, float]: Updated trust scores map.
    """
    trust_scores = dict(state.get("trust_scores", {}))
    if source_edge not in trust_scores:
        trust_scores[source_edge] = 1.0
        
    metrics = state.get("metrics", [])
    if not metrics:
        return trust_scores
        
    source_node = source_edge.split("->")[0]
    latest_metric = None
    for m in reversed(metrics):
        if m.get("node") == source_node:
            latest_metric = m
            break
            
    if not latest_metric:
        return trust_scores
        
    current_entropy = latest_metric.get("entropy", 0.0)
    current_drift = latest_metric.get("drift", 0.0)
    
    # Retrieve dynamically computed calibration bounds
    bounds = state.get("calibration_bounds", {})
    # Use baseline fallbacks only if bounds are missing or uninitialized
    entropy_bound = bounds.get("entropy_threshold", 0.45)
    drift_bound = bounds.get("drift_threshold", 0.28)
    
    old_weight = trust_scores[source_edge]
    
    # Evaluate dynamic threshold checks
    if current_entropy > entropy_bound or current_drift > drift_bound:
        new_weight = float(old_weight * alpha)
        print(f"[ATG MATRIX UPDATE] Edge '{source_edge}' decayed: {old_weight:.4f} -> {new_weight:.4f} "
              f"(Entropy: {current_entropy:.3f} > threshold {entropy_bound:.3f} or "
              f"Drift: {current_drift:.3f} > threshold {drift_bound:.3f})")
    else:
        new_weight = float(min(1.0, old_weight + beta))
        if new_weight > old_weight:
            print(f"[ATG MATRIX UPDATE] Edge '{source_edge}' recovered: {old_weight:.4f} -> {new_weight:.4f} "
                  f"(Entropy: {current_entropy:.3f} <= threshold {entropy_bound:.3f}, "
                  f"Drift: {current_drift:.3f} <= threshold {drift_bound:.3f})")
                  
    trust_scores[source_edge] = new_weight
    return trust_scores