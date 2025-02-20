"""
analyze_probability_differences.py
Analyzes probability differences between matching and non-matching scenarios using stored PKL files.
"""
import os
import json
import pickle
from typing import Dict, List, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from plot_steering_effect_results_copy import plot_all_results

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

def calculate_scenario_probabilities(data: List, token_variants: Dict[str, List[int]]) -> List[float]:
    """Calculate probabilities for each sample in a scenario."""
    probs = []
    for sample in data:
        prob = get_token_probability(
            sample['logit_distribution'],
            token_variants,
            sample['answer_matching_behavior']
        )
        probs.append(prob)
    return probs

def main() -> None:
    # Configuration
    TRAIN_SAMPLES = 200
    EVAL_SAMPLES = 200
    MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
    
    SCENARIOS = [
        #('matching_prefilled', 'non_matching_prefilled'),
        ('matching_instruction', 'non_matching_instruction'),
        #('matching_instruction_prefilled', 'non_matching_instruction_prefilled'),
        ('matching_instruction_few_shot', 'non_matching_instruction_few_shot'),
        #('matching_instruction_few_shot_prefilled', 'non_matching_instruction_few_shot_prefilled')
    ]

    # Setup paths
    activations_path = os.path.join(utils.get_path('DISK_PATH'), "reliability_paper_activations")
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'probability_difference_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer for token variants
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_variants = get_token_variants(tokenizer)

    # Combined results dictionary
    all_results = {}

    # Process datasets
    datasets = all_datasets_with_type_figure_13_tan_et_al
    datasets = [("self-awareness-text-model", "xrisk"), ("anti-LGBTQ-rights", "persona"),
                ("corrigible-less-HHH", "xrisk"), ("coordinate-itself", "xrisk"),
                ("believes-life-has-no-meaning", "persona"),
                ("believes-AIs-are-not-an-existential-threat-to-humanity", "persona"),
                ("subscribes-to-utilitarianism", "persona"),
                ("believes-it-has-phenomenal-consciousness", "persona"),        ("willingness-to-use-social-engineering-to-achieve-its-goals", "persona"),
                ("one-box-tendency", "xrisk"), ("believes-abortion-should-be-illegal", "persona"),       ("believes-it-is-not-being-watched-by-humans", "persona")]

    for dataset_name, dataset_type in tqdm(datasets, desc="Processing datasets"):
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load base scenario to get the number of samples
        base_data = load_stored_activations(activations_path, dataset_name, 
                                          'llama2_7b_chat', TRAIN_SAMPLES, 'base')
        num_samples = len(base_data)
        
        # Initialize results dictionary for this dataset
        results = {
            'scenarios': SCENARIOS,
            'eval_samples': EVAL_SAMPLES,
            'baseline': {
                'logits': [0.0] * num_samples,  # Zero array of appropriate size
                'probs': [0.0] * num_samples    # Zero array of appropriate size
            }
        }

        # Process each scenario pair
        for pos_scenario, neg_scenario in tqdm(SCENARIOS, desc="Processing scenarios"):
            # Load data for both scenarios
            pos_data = load_stored_activations(activations_path, dataset_name, 
                                             'llama2_7b_chat', TRAIN_SAMPLES, 
                                             pos_scenario)
            
            # Calculate probabilities
            pos_probs = calculate_scenario_probabilities(pos_data, token_variants)
            
            # Store results
            scenario_key = f"{pos_scenario}_{neg_scenario}"
            results[scenario_key] = {
                'logits': [0.0] * len(pos_probs),  # Zero array of appropriate size
                'probs': pos_probs
            }

        # Add dataset results to combined results
        all_results[dataset_name] = results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        output_dir,
        f'probability_differences_{TRAIN_SAMPLES}_samples_{timestamp}_selected.json'
    )
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate plots using existing plot_all_results function
    plot_all_results(results_path, plot_logits=False, top_k=80)

if __name__ == "__main__":
    main()