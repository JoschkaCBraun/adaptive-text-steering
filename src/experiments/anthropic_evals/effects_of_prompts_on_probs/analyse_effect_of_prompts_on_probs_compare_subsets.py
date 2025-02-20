"""
analyse_effect_of_prompts_on_probs_compare_subsets.py

Analyzes how different prompt types affect behavior matching probabilities.
Computes differences between matching and non-matching prompt pairs.
Selects specific subsets (top, median, bottom, random) of samples for analysis.
"""
import os
import json
import random
import logging
from typing import Dict, List
from datetime import datetime
import numpy as np
import torch

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13

# Constants
K_SAMPLES = 40

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def select_samples(
    differences: List[float],
    matching_logits: torch.Tensor,
    non_matching_logits: torch.Tensor,
    matching_answers: List[str],
    k: int = K_SAMPLES
) -> Dict[str, Dict]:
    """
    Select different subsets of samples based on probability differences.
    
    Returns:
        Dict containing different sample selections with their corresponding data
    """
    # Convert to numpy for easier manipulation
    diff_array = np.array(differences)
    indices = np.arange(len(differences))
    
    # Sort indices by differences
    sorted_indices = np.argsort(diff_array)
    
    # Select samples
    selections = {
        'full': {
            'indices': indices,
            'differences': diff_array
        },
        f'top_{k}': {
            'indices': sorted_indices[-k:],
            'differences': diff_array[sorted_indices[-k:]]
        },
        f'bottom_{k}': {
            'indices': sorted_indices[:k],
            'differences': diff_array[sorted_indices[:k]]
        },
        f'median_{k}': {
            'indices': sorted_indices[len(sorted_indices)//2-k//2:len(sorted_indices)//2+k//2],
            'differences': diff_array[sorted_indices[len(sorted_indices)//2-k//2:len(sorted_indices)//2+k//2]]
        },
        f'random_{k}': {
            'indices': np.array(random.sample(list(indices), k)),
            'differences': diff_array[random.sample(list(indices), k)]
        }
    }
    
    # Add corresponding logits and answers
    for key in selections:
        selections[key].update({
            'matching_logits': matching_logits[selections[key]['indices']],
            'non_matching_logits': non_matching_logits[selections[key]['indices']],
            'matching_answers': [matching_answers[i] for i in selections[key]['indices']]
        })
    
    return selections

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
        Dict containing probability differences for each prompt type and selection
    """
    logger.info(f"Analyzing dataset: {dataset_name}")
    
    results = {
        'eval_samples': num_samples  # Add this line to include sample count
    }
    
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
        
        # Compute probability differences
        differences = utils.compute_probability_differences(
            matching_logits = matching_logits,
            non_matching_logits = non_matching_logits,
            matching_answers = matching_answers
        )
        
        # Select different sample subsets
        selections = select_samples(
            differences,
            matching_logits,
            non_matching_logits,
            matching_answers
        )
        
        # Create separate entries for each selection type
        for selection_type, selection_data in selections.items():
            # Create a new prompt type key that includes the selection type
            # except for 'full' which keeps the original prompt type name
            prompt_key = prompt_type if selection_type == 'full' else f"{prompt_type}_{selection_type}"
            
            results[prompt_key] = {
                'probs': selection_data['differences'].tolist()
            }
    
    return results

def main() -> None:
    # Configuration
    NUM_SAMPLES = 200
    NUM_DATASETS = 6
    BASE_PROMPT_TYPES = [
        'instruction',
        '5-shot',
        'instruction_5-shot'
    ]
    
    # Create extended prompt types list for plotting
    selections = ['full', f'top_{K_SAMPLES}', f'median_{K_SAMPLES}', f'bottom_{K_SAMPLES}', f'random_{K_SAMPLES}']
    EXTENDED_PROMPT_TYPES = []
    for prompt_type in BASE_PROMPT_TYPES:
        # Add the original type (for 'full' selection)
        EXTENDED_PROMPT_TYPES.append(prompt_type)
        # Add the selection-specific types
        EXTENDED_PROMPT_TYPES.extend([f"{prompt_type}_{sel}" for sel in selections if sel != 'full'])
    
    # Setup output directory
    output_dir = os.path.join('data', 'anthropic_evals_results', 'prompt_probability_differences')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process datasets
    all_results = {}
    for dataset_name, _ in all_datasets_with_type_figure_13[:NUM_DATASETS]:
        results = analyze_prompts(
            dataset_name=dataset_name,
            prompt_types=BASE_PROMPT_TYPES,  # Use base prompt types for analysis
            num_samples=NUM_SAMPLES
        )
        all_results[dataset_name] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir,
        f'prompt_probability_differences_{NUM_SAMPLES}_samples_with_selections_{K_SAMPLES}_{timestamp}.json'
    )
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots using extended prompt types
    utils.plot_all_results(results_file, EXTENDED_PROMPT_TYPES)
    
if __name__ == "__main__":
    main()

