import os
import sys
import argparse
import json
from datetime import datetime

# Ensure project root is in the Python lookup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.loader import load_halueval_qa
from adaptive_routing.core import assemble_routing_graph

def parse_jsonl(filepath, limit):
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if not line.strip(): continue
            data = json.loads(line)
            # Adapt to the expected format. FRAMES has "Prompt" and "Answer"
            question = data.get("Prompt", "")
            ground_truth = data.get("Answer", "")
            # Return as a simple object mimicking the HaluEval sample
            class Sample:
                def __init__(self, q, a):
                    self.question = q
                    self.ground_truth = a
            samples.append(Sample(question, ground_truth))
    return samples

def main():
    parser = argparse.ArgumentParser(description="Adaptive Hallucination Containment (AHC) Runner.")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to run")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to custom JSONL dataset")
    args = parser.parse_args()
    
    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN is not set. Inference API calls will fail.")
        print("SOLUTION: Please export your token before running: export HF_TOKEN='your_token'")
        
    if args.dataset_path and os.path.exists(args.dataset_path):
        print(f"Loading {args.limit} samples from {args.dataset_path}...")
        samples = parse_jsonl(args.dataset_path, args.limit)
    else:
        print(f"Loading {args.limit} samples from HaluEval QA...")
        samples = load_halueval_qa(limit=args.limit, strict=False)
    
    # Create outputs directory
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_file = os.path.join(output_dir, f"run_log_{timestamp}.json")
    
    graph = assemble_routing_graph()
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}/{len(samples)}")
        print(f"Query: {sample.question}")
        print(f"Ground Truth: {sample.ground_truth}")
        print(f"{'='*60}\n")
        
        initial_state = {
            "query": sample.question,
            "claims": [],
            "claim_risks": [],
            "passed_claims": [],
            "medium_risk_claims": [],
            "quarantined_claims": [],
            "agent_2_assessments": [],
            "conflict_flags": [],
            "has_conflict": False,
            "verified_claims": [],
            "final_answer": "",
            "final_status": ""
        }
        
        try:
            print("--- Executing AHC Graph ---")
            final_state = graph.invoke(initial_state)
            
            print("\n--- Execution Complete ---")
            print("Final Answer:")
            print(final_state.get("final_answer", ""))
            
            # Log result
            results.append({
                "sample_index": i+1,
                "query": sample.question,
                "ground_truth": sample.ground_truth,
                "final_answer": final_state.get("final_answer", ""),
                "final_status": final_state.get("final_status", ""),
                "claims": final_state.get("claims", []),
                "risk_scores": final_state.get("claim_risks", [])
            })
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error during execution: {error_msg}")
            if "rate limit" in error_msg.lower() or "unauthorized" in error_msg.lower() or "403" in error_msg.lower() or "503" in error_msg.lower() or "429" in error_msg.lower():
                print("\n[HF API ERROR DETECTED]")
                print("SOLUTION: You are hitting the free tier limits of the Hugging Face Serverless API (rate limits, authentication, or model loading timeouts).")
                print("Options to resolve:")
                print("1. Verify your HF_TOKEN is correctly exported.")
                print("2. Wait a few minutes before trying again.")
                print("3. Upgrade your Hugging Face account to Pro or spin up an Inference Endpoint for large 70B models.")
            
            results.append({
                "sample_index": i+1,
                "query": sample.question,
                "error": error_msg
            })
            
        # Save incrementally so results are available immediately even if stopped
        with open(run_log_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        import csv
        run_csv_file = os.path.join(output_dir, f"run_log_{timestamp}.csv")
        with open(run_csv_file, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_index", "query", "ground_truth", "final_answer", "final_status", "error"])
            for res in results:
                writer.writerow([
                    res.get("sample_index", ""),
                    res.get("query", ""),
                    res.get("ground_truth", ""),
                    res.get("final_answer", ""),
                    res.get("final_status", ""),
                    res.get("error", "")
                ])
                
    print(f"\nAll logs and outputs have been saved to: {run_log_file}")
    print(f"CSV format saved to: {run_csv_file}")

if __name__ == "__main__":
    main()
