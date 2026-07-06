# src/metrics/semantic.py
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Singleton instance configuration to avoid reloading models on every step
_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer('BAAI/bge-large-en-v1.5')
    return _EMBEDDER

def calculate_semantic_drift(current_text: str, anchor_text: str) -> dict:
    """
    Computes the cosine distance between two text strings using a local embedding matrix.
    Logs precise processing latency to track computational runtime requirements.
    """
    start_time = time.perf_counter()
    model = get_embedder()
    
    vec1 = model.encode(current_text, convert_to_numpy=True)
    vec2 = model.encode(anchor_text, convert_to_numpy=True)
    
    # Calculate cosine similarity using NumPy vector operations
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    distance = float(1.0 - similarity)
    
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    return {
        "drift": distance,
        "latency_ms": latency_ms
    }