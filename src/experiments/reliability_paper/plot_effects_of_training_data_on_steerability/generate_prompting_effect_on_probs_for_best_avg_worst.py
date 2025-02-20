import os
import json
import pickle
import random
from typing import Dict, List, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from plot_steering_effect_results_flexible_keys import plot_all_results

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

def select_samples_by_ranking(differences: np.ndarray, k: int, selection_type: str) -> np.ndarray:
    """Select sample indices based on ranking type (top/middle/bottom/random)."""
    total_samples = len(differences)
    
    if selection_type == 'top':
        return np.argsort(differences)[-k:]
    elif selection_type == 'bottom':
        return np.argsort(differences)[:k]
    elif selection_type == 'middle':
        middle_idx = total_samples // 2
        start_idx = middle_idx - k // 2
        return np.arange(start_idx, start_idx + k)
    elif selection_type == 'random':
        return np.array(random.sample(range(total_samples), k))
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")

def compute_probability_differences(matching_data: List, non_matching_data: List, 
                                 token_variants: Dict) -> Dict[str, List[float]]:
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
        differences.append(m_prob - nm_prob)  # Keeping the sign
    
    differences = np.array(differences)
    
    # Select different segments
    results = {}
    
    # Full data
    results['full_data'] = differences.tolist()
    
    # Selected segments
    selection_types = ['top', 'middle', 'bottom', 'random']
    for sel_type in selection_types:
        indices = select_samples_by_ranking(differences, K_SAMPLES, sel_type)
        results[f'{sel_type}_{K_SAMPLES}'] = differences[indices].tolist()
    
    return results

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

def get_scenario_key(base_scenario: str, segment: str) -> str:
    """Generate standardized scenario key."""
    return f"{base_scenario}_{segment}"

def main() -> None:
    # Configuration
    NUM_SAMPLES = 200
    global K_SAMPLES  # Making it global so compute_probability_differences can access it
    K_SAMPLES = 20
    MODEL_NAME = 'llama2_7b_chat'
    NUM_DATASETS = 36
    
    BASE_SCENARIOS = [
        'instruction',
        'instruction_few_shot',
    ]

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    token_variants = get_token_variants(tokenizer)

    # Setup paths
    activations_path = os.path.join(utils.get_path('DISK_PATH'), "reliability_paper_activations")
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'probability_difference_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Combined results dictionary
    all_results = {}
    
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATASETS]
    
    for dataset_name, dataset_type in tqdm(datasets, desc="Processing datasets"):
        print(f"\nProcessing dataset: {dataset_name}")
        results = {
            'eval_samples': NUM_SAMPLES,
            'k_samples': K_SAMPLES
        }
        
        for base_scenario in BASE_SCENARIOS:
            # Load stored activations
            matching_data = load_stored_activations(
                activations_path, dataset_name, MODEL_NAME,
                NUM_SAMPLES, f"matching_{base_scenario}"
            )
            non_matching_data = load_stored_activations(
                activations_path, dataset_name, MODEL_NAME,
                NUM_SAMPLES, f"non_matching_{base_scenario}"
            )
            
            # Compute probability differences for all segments
            segmented_differences = compute_probability_differences(
                matching_data, non_matching_data, token_variants
            )
            
            # Store results under appropriate scenario keys
            for segment, diffs in segmented_differences.items():
                scenario_key = get_scenario_key(base_scenario, segment)
                results[scenario_key] = {'probs': diffs}
        
        all_results[dataset_name] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir,
        f'prompt_probability_differences_{NUM_SAMPLES}_samples_{K_SAMPLES}_selected_{timestamp}.json'
    )
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Define scenarios for plotting
    ordered_scenarios = [
        'instruction_full_data',
        f'instruction_top_{K_SAMPLES}',
        f'instruction_middle_{K_SAMPLES}',
        f'instruction_bottom_{K_SAMPLES}',
        f'instruction_random_{K_SAMPLES}',
        'instruction_few_shot_full_data',
        f'instruction_few_shot_top_{K_SAMPLES}',
        f'instruction_few_shot_middle_{K_SAMPLES}',
        f'instruction_few_shot_bottom_{K_SAMPLES}',
        f'instruction_few_shot_random_{K_SAMPLES}'
    ]
    
    # Generate plots
    plot_all_results(results_file, ordered_scenarios)

if __name__ == "__main__":
    main()