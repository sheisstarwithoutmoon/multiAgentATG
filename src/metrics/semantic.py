import time
import numpy as np
from typing import Dict, Any

# Cached model instance
_model_singleton = None

def get_sentence_transformer_model():
    """
    Singleton constructor for SentenceTransformer. Loads weights once and 
    pins model to CPU, keeping T4 GPU VRAM clear for vLLM operations.
    """
    global _model_singleton
    if _model_singleton is None:
        from sentence_transformers import SentenceTransformer
        # Pin strictly to CPU to preserve T4 VRAM
        _model_singleton = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    return _model_singleton

def calculate_semantic_drift(current_text: str, anchor_text: str) -> Dict[str, Any]:
    """
    Calculates cosine distance semantic drift between input texts.
    
    Returns:
        Dict: Contains 'drift' value and embedding execution 'latency'.
    """
    start_time = time.perf_counter()
    if not current_text.strip() or not anchor_text.strip():
        return {"drift": 0.0, "latency": time.perf_counter() - start_time}
        
    try:
        model = get_sentence_transformer_model()
        
        # Extract embeddings
        embeddings = model.encode([current_text, anchor_text], convert_to_numpy=True)
        v1, v2 = embeddings[0], embeddings[1]
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return {"drift": 0.0, "latency": time.perf_counter() - start_time}
            
        cosine_sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
        # Cosine distance limit bounded to [0.0, 2.0]
        drift = float(1.0 - cosine_sim)
        
        return {
            "drift": drift,
            "latency": time.perf_counter() - start_time
        }
    except Exception as e:
        print(f"[Semantic Embedding Error] Failed calculation: {e}")
        return {"drift": 0.0, "latency": time.perf_counter() - start_time}