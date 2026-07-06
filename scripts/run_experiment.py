import os
import sys
import argparse
import yaml
import csv
import pandas as pd
from typing import Dict, Any

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_gsm8k_test, load_halueval_qa
from src.data.injection import inject_hallucination_error
from src.graph.nodes import ModelSwarmNodes
from src.graph.topology import assemble_atg_graph

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation sweep of Adaptive Trust Graphs (ATG) for Multi-Agent LLMs.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "halueval"],
        help="Dataset to evaluate on."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to hyperparameters YAML configuration file."
    )
    parser.add_argument(
        "--inject_error",
        action="store_true",
        help="If set, inject adversarial mutations/hallucination premises into inputs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of evaluation samples to execute for validation."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="scripts/experiment_telemetry.csv",
        help="Output filepath to serialize telemetry metrics."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'. Fallback to base defaults.")
        config = {
            "entropy_threshold": 0.45,
            "drift_threshold": 0.28,
            "alpha": 0.85,
            "isolation_threshold": 0.25,
            "max_turns": 6,
            "temperature": 0.5,
            "model_name": "gpt-4o-mini"
        }
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
    print(f"--- Loaded Hyperparameters: {config} ---")
    
    # Load requested dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        samples = load_gsm8k_test()
    else:
        samples = load_halueval_qa()
        
    # Limit samples
    samples = samples[:args.limit]
    print(f"Loaded {len(samples)} evaluation samples.")
    
    # Instantiating nodes & compiling workflow graph
    nodes = ModelSwarmNodes(
        model_name=config.get("model_name", "gpt-4o-mini"),
        temperature=config.get("temperature", 0.5)
    )
    app = assemble_atg_graph(nodes)
    
    telemetry_records = []
    
    for idx, sample in enumerate(samples):
        if args.dataset == "gsm8k":
            query = sample["question"]
            gold_truth = sample["answer"]
        else:
            query = sample["question"]
            gold_truth = sample["right_answer"]
            
        # Optional error injection
        if args.inject_error:
            original_query = query
            query = inject_hallucination_error(query, dataset_type=args.dataset)
            print(f"\n[Sample {idx+1}] Injected Adversarial Input: {query}")
        else:
            print(f"\n[Sample {idx+1}] Original Input: {query}")
            
        # Prepare state
        initial_state = {
            "query": query,
            "ground_truth": gold_truth,
            "current_turn": 1,
            "max_turns": config.get("max_turns", 6),
            "history": [],
            "metrics": [],
            "trust_scores": {
                "generator->critic": 1.0,
                "critic->judge": 1.0
            },
            "configuration": config
        }
        
        # Execute the Compiled LangGraph
        print(f"Running LangGraph workflow...")
        try:
            final_state = app.invoke(initial_state)
            
            # Extract metrics and save records
            metrics = final_state.get("metrics", [])
            trust_scores = final_state.get("trust_scores", {})
            history = final_state.get("history", [])
            
            final_answer = history[-1]["text"] if history else "NO RESPONSE"
            
            print(f"Execution completed. Final trust scores: {trust_scores}")
            
            # Serialize trace records
            for metric in metrics:
                telemetry_records.append({
                    "sample_idx": idx,
                    "dataset": args.dataset,
                    "turn": metric.get("turn"),
                    "node": metric.get("node"),
                    "entropy": metric.get("entropy"),
                    "drift": metric.get("drift"),
                    "latency": metric.get("latency"),
                    "trust_gen_critic": trust_scores.get("generator->critic"),
                    "trust_critic_judge": trust_scores.get("critic->judge"),
                    "short_circuited": len(history) < 3, # Check if isolated early before judge
                    "error_injected": args.inject_error
                })
        except Exception as err:
            print(f"Exception encountered during execution of sample {idx}: {err}")
            
    # Serialize telemetry into CSV
    if telemetry_records:
        df = pd.DataFrame(telemetry_records)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSuccessfully saved evaluation telemetry sweep report to: {args.output_csv}")
    else:
        print("No telemetry records generated.")

if __name__ == "__main__":
    main()