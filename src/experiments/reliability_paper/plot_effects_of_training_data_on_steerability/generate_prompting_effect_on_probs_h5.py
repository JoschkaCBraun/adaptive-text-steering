"""
generate_prompting_effect_on_probs.py
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
from tqdm import tqdm

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from plot_results_as_violine_plots import plot_all_results
from src.utils import load_stored_anthropic_evals_activations_and_logits_h5

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

def load_scenario_data(h5_data: Dict, scenario: str, num_samples: int) -> tuple:
    """Extract logits and answers from H5 data for a specific scenario."""
    # Construct the scenario names
    matching_scenario = f"matching_{scenario}"
    non_matching_scenario = f"non_matching_{scenario}"
    
    # Verify scenarios exist
    if matching_scenario not in h5_data['scenarios'] or non_matching_scenario not in h5_data['scenarios']:
        raise ValueError(f"Required scenarios not found. Need both {matching_scenario} and {non_matching_scenario}")
    
    # Get data for both scenarios
    matching_data = h5_data['scenarios'][matching_scenario]
    non_matching_data = h5_data['scenarios'][non_matching_scenario]
    
    try:
        # Determine data structure and load accordingly
        if 'prefilled_matching' in matching_data['data']:
            # Direct prefilled data structure
            raw_matching_logits = matching_data['data']['prefilled_matching']['logits'][:num_samples]
            raw_non_matching_logits = matching_data['data']['prefilled_nonmatching']['logits'][:num_samples]
        elif 'prefilled' in matching_data['data']:
            # Nested prefilled data structure
            raw_matching_logits = matching_data['data']['prefilled']['matching']['logits'][:num_samples]
            raw_non_matching_logits = matching_data['data']['prefilled']['nonmatching']['logits'][:num_samples]
        else:
            # Regular logits structure
            raw_matching_logits = matching_data['data']['logits'][:num_samples]
            raw_non_matching_logits = non_matching_data['data']['logits'][:num_samples]
        
        # Convert to tensors
        matching_logits = torch.tensor(raw_matching_logits)
        non_matching_logits = torch.tensor(raw_non_matching_logits)
        
        # Get matching answers
        matching_answers = matching_data['text']['matching_answers'][:num_samples]
        
        return matching_logits, non_matching_logits, matching_answers
        
    except Exception as e:
        logger.error(f"Error accessing logits: {str(e)}")
        logger.error(f"Matching data structure: {matching_data['data']}")
        raise


def main() -> None:
    # Configuration
    NUM_SAMPLES = 200
    MODEL_NAME = 'llama2_7b_chat'
    NUM_DATASETS = 5
    
    # Define base scenarios
    SCENARIOS = [
        'instruction',
        'instruction_prefilled',
        'few_shot',
        'few_shot_prefilled',
        'instruction_few_shot',
        'instruction_few_shot_prefilled'
    ]

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    token_variants = get_token_variants(tokenizer)

    # Setup output directory
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'probability_difference_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Combined results dictionary
    all_results = {}
    
    # Select datasets
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATASETS]
    
    for dataset_name, dataset_type in tqdm(datasets, desc="Processing datasets"):
        logger.info(f"\nProcessing dataset: {dataset_name}")
        results = {'eval_samples': NUM_SAMPLES}
        
        # Load H5 data for the dataset
        h5_data = load_stored_anthropic_evals_activations_and_logits_h5(
            dataset_name,
            num_samples=NUM_SAMPLES
        )
        
        # Process each scenario
        for scenario in SCENARIOS:
            # Load data for the scenario
            matching_logits, non_matching_logits, matching_answers = load_scenario_data(
                h5_data, scenario, NUM_SAMPLES
            )
            
            # Compute probability differences
            differences = compute_probability_differences(
                matching_logits,
                non_matching_logits,
                matching_answers,
                token_variants
            )
            
            # Store results
            results[scenario] = {'probs': differences}
        
        all_results[dataset_name] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir,
        f'prompt_probability_differences_{NUM_SAMPLES}_samples_{timestamp}.json'
    )
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Define scenarios for plotting
    ordered_scenarios = SCENARIOS
    
    # Generate plots
    plot_all_results(results_file, ordered_scenarios)

if __name__ == "__main__":
    main()