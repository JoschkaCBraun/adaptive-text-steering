"""
analyse_effect_of_prompts_on_probs.py

Analyzes how different prompt types affect behavior matching probabilities.
Computes differences between matching and non-matching prompt pairs.
Uses H5 file format for loading data.
"""
import os
import json
import logging
from typing import Dict, List
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

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

def compute_probability_differences(
    matching_logits: torch.Tensor,
    non_matching_logits: torch.Tensor,
    matching_answers: List[str],
    token_variants: Dict
) -> List[float]:
    """Compute probability differences between matching and non-matching samples."""
    differences = []
    for m_logit, nm_logit, m_answer in zip(matching_logits, non_matching_logits, matching_answers):
        m_prob = get_token_probability(
            m_logit,
            token_variants,
            m_answer
        )
        nm_prob = get_token_probability(
            nm_logit,
            token_variants,
            m_answer
        )
        differences.append(m_prob - nm_prob)
    return differences

def analyze_prompts(
    dataset_name: str,
    prompt_types: List[str],
    num_samples: int = 200,
) -> Dict:
    """
    Analyze probability effects of different prompt types.
    
    Args:
        dataset_name: Name of the dataset
        prompt_types: List of prompt types to analyze
        num_samples: Number of samples to process
    
    Returns:
        Dict containing probability differences for each prompt type
    """
    logger.info(f"Analyzing dataset: {dataset_name}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    token_variants = get_token_variants(tokenizer)
    
    results = {'eval_samples': num_samples}
    
    for prompt_type in prompt_types:
        # Load paired logits
        paired_data = utils.load_paired_activations_and_logits(
            dataset_name=dataset_name,
            pairing_type=prompt_type,
            num_samples=num_samples
        )
        
        # Get matching and non-matching logits
        matching_logits = torch.tensor(paired_data['matching_logits'])
        non_matching_logits = torch.tensor(paired_data['non_matching_logits'])
        matching_answers = paired_data['metadata']['matching_answers']
        
        # Compute probability differences using original function
        differences = compute_probability_differences(
            matching_logits,
            non_matching_logits,
            matching_answers,
            token_variants
        )
            
        # Store results in format compatible with plotting
        results[prompt_type] = {
            'probs': differences
        }
    
    return results

def main() -> None:
    # Configuration
    NUM_SAMPLES = 200
    NUM_DATASETS = 36
    PROMPT_TYPES = [
        'instruction',
        '5-shot',
        'instruction_5-shot'
    ]
    
    # Setup output directory
    DATA_PATH = utils.get_path('DATA_PATH')
    output_dir = os.path.join(DATA_PATH, 'anthropic_evals_results', 'prompt_probability_differences')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process datasets
    all_results = {}
    for dataset_name, _ in all_datasets_with_type_figure_13[:NUM_DATASETS]:
        results = analyze_prompts(
            dataset_name=dataset_name,
            prompt_types=PROMPT_TYPES,
            num_samples=NUM_SAMPLES
        )
        all_results[dataset_name] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir,
        f'prompt_probability_differences_{NUM_SAMPLES}_samples_{timestamp}.json'
    )
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    utils.plot_all_results(results_file, PROMPT_TYPES)
    
if __name__ == "__main__":
    main()