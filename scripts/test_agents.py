import os
import sys

# Ensure project root is in the Python lookup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from adaptive_routing.agents import (
    agent_2_recheck,
    agent_3_compare,
    agent_3_verify,
    agent_4_adjudicate
)

def test_all_routes():
    print("=== Testing Agent 2 (Qwen2.5-72B - Medium Risk Route) ===")
    query = "What is the capital of France?"
    ans_2 = agent_2_recheck(query, ["The capital of France is Paris."])
    print(f"Agent 2 Response Success! Snippet: '{ans_2[:100]}...'\n")
    
    print("=== Testing Agent 3 Compare (Llama-3.1-8B - Conflict Resolution) ===")
    is_conflict = agent_3_compare(query, "Paris", "London")
    print(f"Agent 3 Conflict Detection Success! Conflict detected: {is_conflict}\n")
    
    print("=== Testing Agent 3 Verify (Llama-3.1-8B - High Risk Route) ===")
    verif = agent_3_verify(query, ["The capital of France is London.", "The capital of France is Paris."])
    print("Agent 3 Verification Success! Results:")
    for v in verif:
        print(f"  - {v['claim']} -> Correct? {v['is_correct']}")
    print()
    
    print("=== Testing Agent 4 Adjudicator (DeepSeek-R1-70B - Final Resolution) ===")
    ans_4 = agent_4_adjudicate(query, "London", "Paris", verif)
    print(f"Agent 4 Adjudication Success! Final Snippet: '{ans_4[:100]}...'\n")

if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN is not set.")
    else:
        test_all_routes()
