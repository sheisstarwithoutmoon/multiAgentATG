import os
from typing import List, Dict, Any
from datasets import load_dataset

def load_gsm8k_test() -> List[Dict[str, Any]]:
    """
    Loads the standard GSM8K test split.
    
    Returns:
        List[Dict[str, Any]]: List of test samples with keys 'question' and 'answer'.
    """
    try:
        # Load gsm8k dataset test split
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = []
        for row in dataset:
            samples.append({
                "question": row["question"],
                "answer": row["answer"]
            })
        return samples
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        # Return fallback mock example if offline/dataset unavailable
        return [
            {
                "question": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much money did she earn?",
                "answer": "Weng earns $12 an hour. Babysitting for 5 hours means she earns 12 * 5 = $60.\n#### 60"
            }
        ]

def load_halueval_qa() -> List[Dict[str, Any]]:
    """
    Loads the HaluEval QA subset from Hugging Face datasets.
    
    Returns:
        List[Dict[str, Any]]: List of QA subset containing query, ground_truth, and hallucinated answers.
    """
    try:
        # Load halueval dataset qa split or general split
        # HaluEval contains subsets: qa, summarization, dialogue, etc.
        dataset = load_dataset("foolwood/halueval", "qa", split="data")
        samples = []
        for row in dataset:
            samples.append({
                "question": row["question"],
                "right_answer": row["right_answer"],
                "hallucinated_answer": row["hallucinated_answer"],
                "knowledge": row.get("knowledge", "")
            })
        return samples
    except Exception as e:
        print(f"Error loading HaluEval QA dataset: {e}")
        # Return fallback mock example if offline
        return [
            {
                "question": "Who was the director of the movie Inception?",
                "right_answer": "Christopher Nolan",
                "hallucinated_answer": "Steven Spielberg",
                "knowledge": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan."
            }
        ]
