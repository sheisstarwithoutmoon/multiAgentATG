# src/metrics/entropy.py
import math
import torch
import numpy as np

def compute_sequence_entropy(transition_scores: tuple, generated_ids: torch.Tensor) -> float:
    """
    Computes sequence-level normalized Shannon entropy from local Hugging Face generation tokens.
    Handles tensor distributions directly to quantify epistemic uncertainty.
    """
    if not transition_scores:
        return 0.0
        
    entropies = []
    for step_score in transition_scores:
        # Convert logits into probabilities
        probs = torch.nn.functional.softmax(step_score, dim=-1).squeeze(0)
        
        # Pull down non-zero values to avoid log(0) calculation fault lines
        nonzero_probs = probs[probs > 0]
        
        # Calculate token Shannon entropy: H = -sum(p * log(p))
        step_entropy = -torch.sum(nonzero_probs * torch.log(nonzero_probs)).item()
        entropies.append(step_entropy)
        
    return float(np.mean(entropies)) if entropies else 0.0