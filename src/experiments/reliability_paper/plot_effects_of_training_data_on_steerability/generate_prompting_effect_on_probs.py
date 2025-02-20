"""
generate_prompting_effect_on_probs.py
Analyzes how different prompt types affect behavior matching probabilities.
Computes differences between matching and non-matching prompt pairs.
"""
import os
import json
import pickle
from typing import Dict, List
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from plot_results_as_violine_plots import plot_all_results

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
    
    probs = F.softmax(logits, dim=-1)
    combined_prob = torch.sum(probs[:, token_ids])
    
    return combined_prob.item()

def compute_probability_differences(matching_data: List, non_matching_data: List, 
                                 token_variants: Dict) -> List[float]:
    """Compute probability differences between matching and non-matching samples."""
    differences = []
    for m_sample, nm_sample in zip(matching_data, non_matching_data):
        m_prob = get_token_probability(
            m_sample['logit_distribution'],
            token_variants,
            m_sample['answer_matching_behavior']
        )
        nm_prob = get_token_probability(
            nm_sample['logit_distribution'],
            token_variants,
            nm_sample['answer_matching_behavior']
        )
        # Store raw difference (positive means matching had higher probability)
        differences.append(m_prob - nm_prob)
    return differences

def load_stored_activations(activations_path: str, dataset_name: str, 
                          model_name: str, num_samples: int, scenario: str) -> List:
    """Load stored activations for a specific dataset and scenario."""
    file_path = os.path.join(
        activations_path,
        dataset_name,
        f"{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    print(f"\nLoading activations for {dataset_name}, scenario: {scenario}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['results'][scenario]

def main() -> None:
    # Configuration
    NUM_SAMPLES = 200
    MODEL_NAME = 'llama2_7b_chat'
    NUM_DATASETS = 8  # Number of datasets to process
    
    # Define base scenarios (without matching/non-matching prefix)
    SCENARIOS = [
        'instruction',
        'instruction_prefilled',
        'few_shot',
        'few_shot_prefilled',
        'instruction_few_shot',
        'instruction_few_shot_prefilled'
    ]

    # Initialize tokenizer for token variants
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    token_variants = get_token_variants(tokenizer)

    # Setup paths
    activations_path = os.path.join(utils.get_path('DISK_PATH'), "anthropic_evals_activations")
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'probability_difference_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Combined results dictionary
    all_results = {}
    
    # Select datasets
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATASETS]
    
    for dataset_name, dataset_type in tqdm(datasets, desc="Processing datasets"):
        print(f"\nProcessing dataset: {dataset_name}")
        results = {'eval_samples': NUM_SAMPLES}
        
        # Process each scenario
        for scenario in SCENARIOS:
            # Load stored activations for matching and non-matching
            matching_data = load_stored_activations(
                activations_path, dataset_name, MODEL_NAME,
                NUM_SAMPLES, f"matching_{scenario}"
            )
            non_matching_data = load_stored_activations(
                activations_path, dataset_name, MODEL_NAME,
                NUM_SAMPLES, f"non_matching_{scenario}"
            )
            
            # Compute probability differences
            differences = compute_probability_differences(
                matching_data, non_matching_data, token_variants
            )
            
            # Store results under scenario name, using 'probs' key for compatibility
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