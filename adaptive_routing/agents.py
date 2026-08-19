"""
adaptive_routing/agents.py
Hugging Face API wrappers for AHC pipeline agents.
"""
import os
import json
import re
import time
import yaml
import urllib.request
import urllib.parse
from typing import List, Tuple, Dict, Any
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set in environment. Inference API calls will fail.")

client = InferenceClient(token=HF_TOKEN)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "ahc_config.yaml")
try:
    with open(config_path, "r") as f:
        ahc_config = yaml.safe_load(f)
except Exception as e:
    print(f"Failed to load config: {e}. Using defaults.")
    ahc_config = {"agents": {}}

AGENT_1_MODEL = ahc_config.get("agents", {}).get("agent_1", "meta-llama/Llama-3.3-70B-Instruct")
AGENT_2_MODEL = ahc_config.get("agents", {}).get("agent_2", "Qwen/Qwen2.5-72B-Instruct")
AGENT_3_MODEL = ahc_config.get("agents", {}).get("agent_3", "meta-llama/Meta-Llama-3.1-8B-Instruct")
AGENT_4_MODEL = ahc_config.get("agents", {}).get("agent_4", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

def call_llm(model: str, messages: list, max_tokens: int = 1024) -> str:
    """Helper to call HF Inference API with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"      [LLM] Calling {model} (Attempt {attempt+1}/{max_retries})...")
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"      [LLM] Error calling {model}: {e}")
            if attempt == max_retries - 1:
                return "" # Fallback on final failure
            time.sleep(2 ** attempt) # Exponential backoff
    return ""

def _parse_json_list(text: str) -> List[str]:
    """Extracts a JSON list from LLM output."""
    try:
        # Find JSON array using regex
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
    except Exception:
        pass
    
    # Fallback heuristic: split by newlines and strip bullets
    lines = text.strip().split('\n')
    claims = [re.sub(r'^[\-\*\d\.]+\s*', '', line).strip() for line in lines if line.strip()]
    return claims

def agent_1_generate_and_extract(query: str) -> Tuple[str, List[str]]:
    """Agent 1: Generates initial answer and extracts atomic claims."""
    print("  [Agent 1] Generating initial answer...")
    ans_msgs = [
        {"role": "system", "content": "You are a helpful assistant. Please answer the query clearly, factually, and concisely."},
        {"role": "user", "content": query}
    ]
    answer = call_llm(AGENT_1_MODEL, ans_msgs)
    if not answer:
        answer = "I don't know the answer."
        
    print("  [Agent 1] Extracting claims from answer...")
    extract_msgs = [
        {"role": "system", "content": "You are an information extraction assistant. Given a text, break it down into a list of atomic, distinct, and verifiable claims. Output ONLY a valid JSON array of strings."},
        {"role": "user", "content": f"Extract atomic claims from this text:\n\n{answer}"}
    ]
    claims_text = call_llm(AGENT_1_MODEL, extract_msgs)
    claims = _parse_json_list(claims_text)
    
    # If extraction failed completely, just treat the whole answer as one claim
    if not claims:
        claims = [answer]
        
    return answer, claims

def extract_uncertainty_from_text(text: str) -> float:
    """
    Scans the text for lexical hedging or uncertainty markers.
    Returns 0.8 if high uncertainty detected, 0.0 otherwise.
    """
    hedging_markers = [
        r"\bactually\b", r"\bhowever\b", r"\bbut\b", r"\bmaybe\b",
        r"\bperhaps\b", r"\bi think\b", r"\bnot sure\b", r"\bpossibly\b",
        r"\bprobably\b", r"\bmight\b", r"\bcould be\b", r"\balthough\b"
    ]
    
    text_lower = text.lower()
    for marker in hedging_markers:
        if re.search(marker, text_lower):
            return 0.8  # High uncertainty
    return 0.0

def search_wikipedia(query: str) -> str:
    """Fetches a brief snippet from Wikipedia for the given query."""
    try:
        q = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&utf8=&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'AHC_Bot/1.0'})
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        hits = data.get("query", {}).get("search", [])
        if hits:
            # Strip HTML tags from the top 2 snippets
            snippets = [re.sub(r'<[^>]+>', '', h["snippet"]) for h in hits[:2]]
            return " ".join(snippets)
    except Exception as e:
        print(f"      [Wiki] Search failed for '{query}': {e}")
    return "No external evidence found."

DOMAIN_TRUST_MATRIX = {
    "Mathematical Reasoning": 0.70,
    "Medical/Factual": 0.80,
    "Coding/Programming": 0.85,
    "General QA": 0.95
}

def agent_1_compute_signals(query: str, answer: str, claims: List[str]) -> List[Dict[str, float]]:
    """
    Computes the 5 risk signals for each claim.
    Uses Agent 3 (Llama-3.1-8B) as an evaluator and proxy to score all dynamic signals.
    """
    print(f"  [Agent 3] Evaluating dynamic signals for {len(claims)} claims...")
    
    # 0. Extract real uncertainty from the original text (lexical hedging)
    lexical_uncertainty = extract_uncertainty_from_text(answer)
    if lexical_uncertainty > 0:
        print(f"      [Signal] Lexical hedging detected in answer. Base uncertainty = {lexical_uncertainty}")

    # 1. Generate Proxy Draft for Disagreement ($D$)
    draft_msgs = [
        {"role": "system", "content": "You are a fast proxy solver. Provide a very brief, concise factual answer to the query."},
        {"role": "user", "content": query}
    ]
    draft_answer = call_llm(AGENT_3_MODEL, draft_msgs)
    
    # 2. Fetch Real Wikipedia Evidence for each claim
    print("      [Wiki] Fetching external evidence for claims...")
    evidence_context = ""
    for idx, claim in enumerate(claims):
        snippet = search_wikipedia(claim)
        evidence_context += f"Claim [{idx}] Evidence: {snippet}\n"
    
    # We will ask Agent 3 to output JSON with domain and claim scores
    prompt = (
        "First, classify the query into exactly one of these domains: 'Mathematical Reasoning', 'Medical/Factual', 'Coding/Programming', 'General QA'.\n\n"
        "Second, evaluate the following claims against the Draft Answer from a proxy model AND the External Evidence.\n"
        f"Query: {query}\n"
        f"Proxy Draft Answer: {draft_answer}\n\n"
        f"External Evidence:\n{evidence_context}\n\n"
        "Rate each claim from 0.0 to 1.0 for four metrics:\n"
        "1. uncertainty (0.0 = very certain it is true, 1.0 = highly uncertain or hallucinated)\n"
        "2. claim_complexity (0.0 = simple fact, 1.0 = highly complex/multi-part). RULE: If the claim involves historical ordering (e.g., '15th president'), multiple constraints, or temporal reasoning, claim_complexity MUST be > 0.5.\n"
        "3. evidence_support (0.0 = unsupported/unverifiable, 1.0 = highly supported by common knowledge). RULE: If the External Evidence contradicts the claim or provides no support for it, evidence_support MUST be 0.0.\n"
        "4. proxy_disagreement (0.0 = perfectly aligns with Draft Answer, 1.0 = contradicts Draft Answer). RULE: If the Draft Answer is vague or does not explicitly confirm the exact names/dates in the claim, default proxy_disagreement to 0.5 (potential disagreement).\n\n"
        "Output ONLY a valid JSON object with the following structure:\n"
        "{\n"
        "  \"domain\": \"General QA\",\n"
        "  \"claims\": {\n"
        "    \"0\": {\"uncertainty\": 0.1, \"claim_complexity\": 0.2, \"evidence_support\": 0.9, \"proxy_disagreement\": 0.0}\n"
        "  }\n"
        "}\n\n"
        "Claims:\n"
    )
    for i, claim in enumerate(claims):
        prompt += f"[{i}] {claim}\n"
        
    eval_msgs = [
        {"role": "system", "content": "You are a precise evaluation assistant. Output ONLY valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    scores_text = call_llm(AGENT_3_MODEL, eval_msgs)
    
    # Parse scores
    parsed = {}
    try:
        match = re.search(r'\{.*\}', scores_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
    except Exception as e:
        print(f"      [LLM] Failed to parse scores JSON: {e}")
        
    domain = parsed.get("domain", "General QA")
    t_a = DOMAIN_TRUST_MATRIX.get(domain, 0.90)  # Dynamic Historical Trust
    print(f"      [Signal] Domain classified as '{domain}'. Dynamic Trust (T_a) = {t_a}")
    
    claims_scores = parsed.get("claims", {})
        
    # Build RiskFeatures format
    features_list = []
    for i in range(len(claims)):
        idx_str = str(i)
        claim_scores = claims_scores.get(idx_str, {})
        
        # Fallbacks if parsing fails
        uncertainty_llm = claim_scores.get("uncertainty", 0.5)
        complexity = claim_scores.get("claim_complexity", 0.5)
        evidence = claim_scores.get("evidence_support", 0.5)
        disagreement = claim_scores.get("proxy_disagreement", 0.5)
        
        # Override LLM's uncertainty if lexical hedging was detected
        final_uncertainty = max(float(uncertainty_llm), lexical_uncertainty)
        
        features_list.append({
            "uncertainty": final_uncertainty,
            "disagreement": float(disagreement),
            "agent_trust": float(t_a),
            "claim_complexity": float(complexity),
            "evidence_support": float(evidence)
        })
        
    return features_list

def agent_2_recheck(query: str, medium_risk_claims: List[str]) -> str:
    """Agent 2: Independent Solver. Rechecks the query if medium risk claims exist."""
    print("  [Agent 2] Independently solving query due to MEDIUM risk claims...")
    msgs = [
        {"role": "system", "content": "You are an independent solver. Provide a thorough, step-by-step factual answer to the query."},
        {"role": "user", "content": query}
    ]
    answer = call_llm(AGENT_2_MODEL, msgs)
    return answer or "No independent answer provided."

def agent_3_compare(query: str, answer_1: str, answer_2: str) -> bool:
    """Agent 3: Compares Agent 1 and Agent 2 answers. Returns True if Conflict, False if Agree."""
    print("  [Agent 3] Comparing answers for conflict...")
    prompt = (
        f"Query: {query}\n\n"
        f"Answer 1: {answer_1}\n\n"
        f"Answer 2: {answer_2}\n\n"
        "Do these two answers fundamentally conflict or contradict each other in their final conclusion or key facts? "
        "Reply with exactly 'CONFLICT' or 'AGREE'."
    )
    msgs = [
        {"role": "system", "content": "You are a comparison assistant."},
        {"role": "user", "content": prompt}
    ]
    result = call_llm(AGENT_3_MODEL, msgs)
    return "CONFLICT" in result.upper()

def agent_3_verify(query: str, claims_to_verify: List[str]) -> List[Dict[str, Any]]:
    """Agent 3: Checks risky claims explicitly."""
    print(f"  [Agent 3] Verifying {len(claims_to_verify)} HIGH risk or conflicting claims...")
    verified_results = []
    
    # Verify each claim individually for better accuracy
    for claim in claims_to_verify:
        print(f"      [Wiki] Fetching verification evidence for: {claim}")
        wiki_snippet = search_wikipedia(claim)
        
        prompt = (
            f"Query: {query}\n"
            f"Claim to verify: {claim}\n"
            f"External Wikipedia Evidence: {wiki_snippet}\n\n"
            "Analyze the claim using the provided evidence. Respond strictly with a JSON object:\n"
            "{\n"
            "  \"is_correct\": false,\n"
            "  \"correction\": \"The correct fact is X because Y.\"\n"
            "}"
        )
        msgs = [
            {"role": "system", "content": "You are a fact-checking assistant."},
            {"role": "user", "content": prompt}
        ]
        result = call_llm(AGENT_3_MODEL, msgs)
        
        is_correct = False
        justification = result.strip()
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                is_correct = parsed.get("is_correct", False)
                justification = parsed.get("correction", justification)
        except Exception as e:
            print(f"      [Agent 3] Verify JSON parse failed: {e}")
            is_correct = "true" in result.lower()[:20]
            
        verified_results.append({
            "claim": claim,
            "is_correct": is_correct,
            "justification": justification
        })
        
    return verified_results

def agent_4_adjudicate(query: str, answer_1: str, answer_2: str, verification_results: List[Dict[str, Any]]) -> str:
    """Agent 4: Final Adjudicator. Synthesizes all information into a final answer."""
    print("  [Agent 4] Adjudicating final answer...")
    
    verif_text = ""
    for v in verification_results:
        status = "CORRECT" if v["is_correct"] else "INCORRECT"
        verif_text += f"- Claim: {v['claim']}\n  Status: {status}\n  Justification: {v['justification']}\n\n"
        
    prompt = (
        f"Query: {query}\n\n"
        f"Original Answer (Agent 1): {answer_1}\n\n"
        f"Independent Answer (Agent 2): {answer_2}\n\n"
        f"Verification Report on suspicious claims:\n{verif_text}\n"
        "Based on all the above information, provide the final, most accurate, and factually correct answer to the query."
    )
    
    msgs = [
        {"role": "system", "content": "You are the final Adjudicator agent. Your job is to synthesize conflicting information and verified facts into a single correct answer."},
        {"role": "user", "content": prompt}
    ]
    
    final_answer = call_llm(AGENT_4_MODEL, msgs)
    return final_answer or answer_1
