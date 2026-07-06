# scripts/plot_results.py
import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure publication-quality academic style parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'savefig.bbox': 'tight'
})

def generate_paper_plots(csv_path="scripts/experiment_telemetry.csv", output_dir="notebooks/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry file not found at {csv_path}. Please run the experiment first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} empirical records from {csv_path} for scientific visualization.")
    
    # Pre-processing data safely
    df['is_correct'] = df['is_correct'].astype(bool)
    df['went_through_recovery'] = df['went_through_recovery'].astype(bool)
    
    # -------------------------------------------------------------------------
    # PLOT 1: Cumulative Task Accuracy (Ablation Comparison Over Sample Window)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    df['cum_acc'] = df['is_correct'].cumsum() / (df.index + 1)
    
    # Segment data to plot the performance difference of recovered paths
    plt.plot(df.index + 1, df['cum_acc'] * 100, label="ATG Core Framework (With Recovery)", color='#1f77b4', lw=2.5)
    
    # Calculate a proxy for what accuracy would be WITHOUT recovery (treating isolated runs as wrong)
    no_recovery_correct = df.apply(lambda r: r['is_correct'] if not r['went_through_recovery'] else False, axis=1)
    df['cum_acc_no_rec'] = no_recovery_correct.cumsum() / (df.index + 1)
    plt.plot(df.index + 1, df['cum_acc_no_rec'] * 100, label="Baseline (Ablated Recovery Zone)", color='#d62728', linestyle='--', lw=2.0)
    
    plt.xlabel("Evaluation Sample Sequence Index")
    plt.ylabel("Cumulative Accuracy (%)")
    plt.title("System Accuracy Bound Under Adversarial Stress")
    plt.legend(frameon=True)
    plt.ylim(0, 105)
    plt.savefig(f"{output_dir}/fig1_cumulative_accuracy.pdf")
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 2: Core Network Topology Latency Window (Overhead Profiling Plot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='went_through_recovery', y='execution_latency_sec', data=df, palette=['#2ca02c', '#ff7f0e'])
    plt.xticks([0, 1], ['Standard Sequence\n(Consensus Path)', 'Triggered Recovery\n(Quarantine Path)'])
    plt.xlabel("Graph Routing Trajectory Execution Pathway")
    plt.ylabel("Total Latency Duration (Seconds)")
    plt.title("End-to-End Latency Profile of Routing Layouts")
    plt.savefig(f"{output_dir}/fig2_latency_distribution.pdf")
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 3: Execution Horizon (Turn Volume vs Accuracy Mapping)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    turn_counts = df.groupby('turns_run')['is_correct'].agg(['count', 'mean']).reset_index()
    turn_counts['accuracy_pct'] = turn_counts['mean'] * 100
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    
    sns.barplot(x='turns_run', y='count', data=turn_counts, ax=ax1, color='#bcbd22', alpha=0.6)
    sns.lineplot(x=ax1.get_xticks(), y=turn_counts['accuracy_pct'], ax=ax2, color='#9467bd', marker='o', lw=2.5)
    
    ax1.set_xlabel("Total Network Conversation Turns Run")
    ax1.set_ylabel("Volume of Processed Test Samples", color='#bcbd22')
    ax2.set_ylabel("Final Core Task Accuracy (%)", color='#9467bd')
    plt.title("System Convergence and Accuracy Across Turn Horizons")
    plt.savefig(f"{output_dir}/fig3_turn_horizon_convergence.pdf")
    plt.close()

    # ----------------─────────────────────────────────────────────────────────
    # PLOT 4: Trust Topology Decay Mapping (Dynamic Edge Decay Trajectory)
    # ----------------─────────────────────────────────────────────────────────
    plt.figure(figsize=(6, 4))
    sample_indices = []
    gc_weights = []
    cj_weights = []
    
    for idx, row in df.iterrows():
        try:
            scores = ast.literal_eval(row['trust_scores'])
            sample_indices.append(idx + 1)
            gc_weights.append(scores.get('generator->critic', 1.0))
            cj_weights.append(scores.get('critic->judge', 1.0))
        except Exception:
            continue
            
    if sample_indices:
        plt.plot(sample_indices, gc_weights, label=r"Edge $w_{Gen \rightarrow Critic}$", color='#17becf', lw=2.0)
        plt.plot(sample_indices, cj_weights, label=r"Edge $w_{Critic \rightarrow Judge}$", color='#e377c2', linestyle=':', lw=2.0)
        plt.xlabel("Evaluation Sample Sequence Index")
        plt.ylabel("Fluid Adjacency Matrix Edge Weights")
        plt.title("Dynamic Adjacency Edge Trajectories Across Evaluation Window")
        plt.ylim(-0.05, 1.05)
        plt.legend(frameon=True)
        plt.savefig(f"{output_dir}/fig4_trust_edge_trajectories.pdf")
        plt.close()

    print(f"Complete paper plotting pipeline successful! Plots saved cleanly to: {output_dir}/")

if __name__ == "__main__":
    generate_paper_plots()