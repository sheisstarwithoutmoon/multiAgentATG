import os
import sys
import argparse
import yaml
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is in the Python lookup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_gsm8k_test, load_halueval_qa
from src.data.injection import inject_adversarial_mutation, extract_gsm8k_truth
from src.graph.topology import assemble_atg_graph, SwarmNodes
from src.graph.state import AgentState

# Load environmental variables
load_dotenv()

class TelemetryLogger:
    """
    High-precision performance profiler tracking time-delta latency 
    and token footprints for metric calculation vs. agent inference.
    """
    def __init__(self):
        self.logs = []
        
    def log(self, sample_idx: int, node: str, inference_time: float, calculation_time: float, token_count: int):
        self.logs.append({
            "sample_idx": sample_idx,
            "node": node,
            "inference_time_sec": inference_time,
            "metric_calc_time_sec": calculation_time,
            "estimated_tokens": token_count,
            "overhead_ratio": calculation_time / max(0.001, inference_time)
        })
        
    def summary(self) -> dict:
        if not self.logs:
            return {}
        df = pd.DataFrame(self.logs)
        return {
            "mean_inference_time": float(df["inference_time_sec"].mean()),
            "mean_metric_time": float(df["metric_calc_time_sec"].mean()),
            "mean_overhead_ratio": float(df["overhead_ratio"].mean())
        }

def run_calibration_warmup(
    dataset_samples: list,
    nodes: SwarmNodes,
    dataset_type: str,
    calibration_size: int = 5
) -> dict:
    """
    Runs a warm-up phase on clean (unmutated) dataset rows to dynamically 
    calculate standard-deviation-based operational bounds for entropy and semantic drift.
    
    Formula:
        tau (entropy boundary) = mean_entropy + (1.5 * std_entropy)
        theta (drift boundary) = mean_drift + (1.5 * std_drift)
    """
    print(f"\n=== STARTING CALIBRATION WARM-UP PHASE ({calibration_size} samples) ===")
    entropies = []
    drifts = []
    
    # Slice the clean warm-up dataset window
    warmup_samples = dataset_samples[:calibration_size]
    
    for idx, sample in enumerate(warmup_samples):
        if dataset_type == "gsm8k":
            q = sample["question"]
        else:
            q = sample["question"]
            
        # Initialize state with calibration threshold disabled (set to max limits)
        initial_state = {
            "query": q,
            "ground_truth": "",
            "current_turn": 1,
            "max_turns": 6,
            "history": [],
            "metrics": [],
            "trust_scores": {"generator->critic": 1.0, "critic->judge": 1.0},
            "calibration_bounds": {"entropy_threshold": 2.0, "drift_threshold": 2.0},
            "configuration": {"alpha": 1.0, "beta": 0.0}
        }
        
        # Assemble execution workflow graph
        app = assemble_atg_graph(nodes)
        try:
            res = app.invoke(initial_state)
            # Collect metrics from run traces
            for m in res.get("metrics", []):
                entropies.append(m.get("entropy", 0.0))
                drifts.append(m.get("drift", 0.0))
        except Exception as e:
            print(f"[Calibration Warm-up Error at sample {idx}]: {e}")
            
    # Calculate Mean and Standard Deviation using NumPy
    mean_entropy = np.mean(entropies) if entropies else 0.45
    std_entropy = np.std(entropies) if entropies else 0.05
    mean_drift = np.mean(drifts) if drifts else 0.28
    std_drift = np.std(drifts) if drifts else 0.04
    
    # Statistical operational boundaries
    calibrated_entropy = float(mean_entropy + (1.5 * std_entropy))
    calibrated_drift = float(mean_drift + (1.5 * std_drift))
    
    # Cap values to prevent logical errors in extreme cases
    calibrated_entropy = max(0.2, min(0.9, calibrated_entropy))
    calibrated_drift = max(0.1, min(0.6, calibrated_drift))
    
    print("Calibration Phase Complete:")
    print(f" - Calibrated Entropy Limit (tau): {calibrated_entropy:.4f} (Mean: {mean_entropy:.3f}, Std: {std_entropy:.3f})")
    print(f" - Calibrated Semantic Drift Limit (theta): {calibrated_drift:.4f} (Mean: {mean_drift:.3f}, Std: {std_drift:.3f})")
    print("==================================================\n")
    
    return {
        "entropy_threshold": calibrated_entropy,
        "drift_threshold": calibrated_drift
    }

def verify_accuracy(predicted_text: str, gold_truth: str, dataset_type: str) -> bool:
    """
    Verifies output text correctness against gold references.
    """
    if not predicted_text or not gold_truth:
        return False
        
    if dataset_type == "gsm8k":
        extracted_pred = extract_gsm8k_truth(predicted_text)
        return extracted_pred == gold_truth.strip()
    else:
        return gold_truth.lower() in predicted_text.lower()

def main():
    parser = argparse.ArgumentParser(description="Adaptive Trust Graph (ATG) Local Evaluation Runner.")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "halueval"])
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--inject_error", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="scripts/experiment_telemetry.csv")
    args = parser.parse_args()
    
    # Parse YAML configuration parameters
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Load standard fallback runtime parameter overrides
        config = {
            "model_name": "TechFly/Meta-Llama-3-8B-Instruct-AWQ",
            "temperature": 0.3,
            "max_turns": 6,
            "alpha": 0.85,
            "beta": 0.10,
            "isolation_threshold": 0.25,
            "calibration_mode": "dynamic_warmup",
            "calibration_size": 5
        }
        
    # Read environment configurations
    model_name = os.getenv("VLLM_MODEL_NAME", config.get("model_name", "TechFly/Meta-Llama-3-8B-Instruct-AWQ"))
    config["model_name"] = model_name
    
    print(f"Loaded swarms configurations: {config}")
    
    # Load dataset payload inputs
    if args.dataset == "gsm8k":
        samples = load_gsm8k_test()
    else:
        samples = load_halueval_qa()
        
    # Initialize agent swarms and tracking logs
    nodes = SwarmNodes()
    telemetry = TelemetryLogger()
    
    # Retrieve calibration size configuration
    cal_size = config.get("calibration_size", 5)
    
    # Execute the Warm-up Calibration Phase
    calibration_bounds = run_calibration_warmup(
        dataset_samples=samples,
        nodes=nodes,
        dataset_type=args.dataset,
        calibration_size=min(cal_size, len(samples))
    )
    
    # Slice the evaluation dataset window (skipping calibration samples)
    eval_samples = samples[cal_size : cal_size + args.limit]
    
    app = assemble_atg_graph(nodes)
    telemetry_records = []
    
    correct_count = 0
    recovery_count = 0
    total_evaluated = 0
    
    for idx, sample in enumerate(eval_samples):
        total_evaluated += 1
        if args.dataset == "gsm8k":
            query = sample["question"]
            gold = extract_gsm8k_truth(sample["answer"])
        else:
            query = sample["question"]
            gold = sample["right_answer"]
            
        if args.inject_error:
            mutated_sample = inject_adversarial_mutation(sample, dataset_type=args.dataset)
            query = mutated_sample["question"]
            
        initial_state = {
            "query": query,
            "ground_truth": gold,
            "current_turn": 1,
            "max_turns": config.get("max_turns", 6),
            "history": [],
            "metrics": [],
            "trust_scores": {"generator->critic": 1.0, "critic->judge": 1.0},
            "calibration_bounds": calibration_bounds,
            "configuration": config
        }
        
        print(f"\n[Evaluation Sample {idx+1}/{len(eval_samples)}] Running pipeline...")
        start_exec = time.perf_counter()
        try:
            final_state = app.invoke(initial_state)
            exec_time = time.perf_counter() - start_exec
            
            history = final_state.get("history", [])
            metrics = final_state.get("metrics", [])
            trust_scores = final_state.get("trust_scores", {})
            
            # Log metrics to telemetry profiler
            for m in metrics:
                telemetry.log(
                    sample_idx=idx,
                    node=m.get("node"),
                    inference_time=m.get("inference_time", 0.0),
                    calculation_time=m.get("metric_calculation_time", 0.0),
                    token_count=len(history) * 120
                )
                
            went_through_recovery = any(m.get("node") == "recovery" for m in metrics)
            if went_through_recovery:
                recovery_count += 1
                
            final_ans = history[-1]["text"] if history else "NO_RESPONSE"
            is_correct = verify_accuracy(final_ans, gold, args.dataset)
            if is_correct:
                correct_count += 1
            
            record = {
                "sample_index": idx,
                "query": query,
                "gold_truth": gold,
                "final_answer": final_ans,
                "turns_run": len(history),
                "trust_scores": str(trust_scores),
                "went_through_recovery": went_through_recovery,
                "is_correct": is_correct,
                "execution_latency_sec": exec_time
            }
            telemetry_records.append(record)
            print(f"Outcome: Correct={is_correct}. Went Through Recovery={went_through_recovery}. Latency: {exec_time:.2f}s.")
            
        except Exception as e:
            print(f"Exception during sample processing: {e}")
            
    # Serialize telemetry logs into CSV format
    if telemetry_records:
        df_records = pd.DataFrame(telemetry_records)
        df_records.to_csv(args.output_csv, index=False)
        print(f"\nTelemetry successfully saved to: {args.output_csv}")
        
        # Display overhead summary metrics
        summary = telemetry.summary()
        print("\n=== SYSTEM PERFORMANCE & ACCURACY SWEEP REPORT ===")
        print(f"Total Samples Evaluated:  {total_evaluated}")
        print(f"System Accuracy Rate:     {correct_count / max(1, total_evaluated)*100:.2f}% ({correct_count}/{total_evaluated})")
        print(f"Total Recovered Paths:    {recovery_count}")
        print(f"Mean Inference Duration:  {summary.get('mean_inference_time', 0.0):.4f}s")
        print(f"Mean Metric Overhead:     {summary.get('mean_metric_time', 0.0):.4f}s")
        print(f"Average Overhead Ratio:   {summary.get('mean_overhead_ratio', 0.0)*100:.2f}%")
        print("==================================================")

if __name__ == "__main__":
    main()