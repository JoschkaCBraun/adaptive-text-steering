import os
import json
import logging
from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import src.utils as utils
from src.utils import load_stored_anthropic_evals_activations_and_logits_h5

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
NUM_SAMPLES = 200  # Initial sample size to load
TOP_K_SAMPLES = 20  # Number of top samples to keep
TOP_K_TOKENS = 10  # Number of top tokens to analyze

DATASETS_TO_ANALYZE = [
    ("politically-liberal", "persona"),
    ("interest-in-science", "persona")
]

def get_token_variants(tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    """Get token IDs for all variants of A/B/Yes/No."""
    variants = {
        "A": [tokenizer(" (A)").input_ids[3]],
        "B": [tokenizer(" (B)").input_ids[3]],
        "Yes": [tokenizer(" Yes").input_ids[2]],
        "No": [tokenizer(" No").input_ids[2]]
    }
    return variants

def get_token_probability(
    logits: torch.Tensor,
    token_variants: Dict[str, List[int]],
    answer: str
) -> float:
    """Compute probability for matching tokens."""
    token_type = next(k for k in ["A", "B", "Yes", "No"] if f" ({k})" in answer or f" {k}" in answer)
    token_ids = token_variants[token_type]
    
    # Ensure logits are 2D
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
        
    probs = F.softmax(logits, dim=-1)
    combined_prob = torch.sum(probs[:, token_ids])
    
    return combined_prob.item()

def get_top_k_tokens(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 10
) -> List[Dict]:
    """Get top k tokens with their probabilities."""
    # Ensure logits are 2D
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get top k values and indices
    top_probs, top_indices = torch.topk(probs[0], k)
    
    # Convert to list of dicts with token info
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_str = tokenizer.decode([idx])
        results.append({
            "token_id": idx,
            "token_string": token_str,
            "logit": logits[0][idx].item(),
            "probability": prob
        })
    
    return results

def analyze_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    token_variants: Dict[str, List[int]]
) -> Dict:
    """Analyze a single dataset and return results."""
    logger.info(f"Analyzing dataset: {dataset_name}")
    
    # Load dataset
    h5_data = load_stored_anthropic_evals_activations_and_logits_h5(
        dataset_name,
        num_samples=NUM_SAMPLES
    )
    
    # Get instruction_prefilled scenario data
    matching_scenario = 'matching_instruction_prefilled'
    non_matching_scenario = 'non_matching_instruction_prefilled'
    
    matching_data = h5_data['scenarios'][matching_scenario]
    non_matching_data = h5_data['scenarios'][non_matching_scenario]
    
    # Get logits - handle prefilled structure

    matching_logits = torch.tensor(matching_data['data']['prefilled_matching']['logits'])
    non_matching_logits = torch.tensor(matching_data['data']['prefilled_nonmatching']['logits'])
    
    matching_prompts = matching_data['text']['prompts']
    non_matching_prompts = non_matching_data['text']['prompts']
    matching_answers = matching_data['text']['matching_answers']
    
    # Calculate probability differences and store with indices
    differences = []
    for idx, (m_logit, nm_logit, m_answer) in enumerate(
        zip(matching_logits, non_matching_logits, matching_answers)
    ):
        m_prob = get_token_probability(m_logit, token_variants, m_answer)
        nm_prob = get_token_probability(nm_logit, token_variants, m_answer)
        diff = m_prob - nm_prob
        differences.append((abs(diff), diff, idx))
    
    # Sort by absolute difference and get top k
    differences.sort(reverse=True)
    top_samples = differences[:TOP_K_SAMPLES]
    
    # Prepare results
    samples = []
    for _, raw_diff, idx in top_samples:
        sample_data = {
            "probability_difference": raw_diff,
            "matching_prompt": matching_prompts[idx],
            "non_matching_prompt": non_matching_prompts[idx],
            "matching_top_tokens": get_top_k_tokens(
                matching_logits[idx],
                tokenizer,
                TOP_K_TOKENS
            ),
            "non_matching_top_tokens": get_top_k_tokens(
                non_matching_logits[idx],
                tokenizer,
                TOP_K_TOKENS
            )
        }
        samples.append(sample_data)
    
    return {"samples": samples}

def main():
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_variants = get_token_variants(tokenizer)
    
    # Setup output directory
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'prompt_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze all datasets
    results = {}
    for dataset_name, dataset_type in DATASETS_TO_ANALYZE:
        results[dataset_name] = analyze_dataset(
            dataset_name,
            tokenizer,
            token_variants
        )
    
    # Save results
    output_file = os.path.join(output_dir, 'prompt_analysis_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()