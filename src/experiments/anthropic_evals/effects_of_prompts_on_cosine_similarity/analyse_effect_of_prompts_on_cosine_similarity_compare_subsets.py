"""
analyse_effect_of_prompts_on_cosine_similarity_compare_subsets.py
"""

import os
import logging
from typing import Dict, List
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13

# Constants
NUM_SAMPLES = 50
NUM_DATASETS = 3
K_SAMPLES = 10

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def compute_similarities_for_subset(
    matching_activations: torch.Tensor,
    non_matching_activations: torch.Tensor,
    indices: np.ndarray,
    device: str
) -> List[float]:
    """Compute cosine similarities for a subset of samples."""
    # Get activations for the subset and convert to list of tensors
    matching_subset = [torch.tensor(row) for row in matching_activations[indices]]
    non_matching_subset = [torch.tensor(row) for row in non_matching_activations[indices]]
    
    # Compute steering vector for this subset
    steering_vector = utils.compute_contrastive_steering_vector(
        matching_subset, 
        non_matching_subset,
        device
    )
    
    # Prepare difference vectors as a list for similarity computation
    difference_vectors = [m - n for m, n in zip(matching_subset, non_matching_subset)]
    
    # Compute individual similarities against the steering vector
    similarities = utils.compute_many_against_one_cosine_similarity(
        difference_vectors,
        steering_vector,
        device
    )
    
    return similarities

def select_samples_and_compute_similarities(
    matching_activations: np.ndarray,
    non_matching_activations: np.ndarray,
    probability_differences: List[float],
    device: str,
    k: int = K_SAMPLES
) -> Dict[str, List[float]]:
    """Select different subsets of samples and compute their similarities."""
    # Convert to numpy for easier manipulation
    diff_array = np.array(probability_differences)
    indices = np.arange(len(probability_differences))
    
    # Sort indices by differences
    sorted_indices = np.argsort(diff_array)
    
    # Define selections
    selections = {
        'all': indices,
        'top': sorted_indices[-k:],
        'middle': sorted_indices[len(sorted_indices)//2-k//2:len(sorted_indices)//2+k//2],
        'bottom': sorted_indices[:k],
        'random': np.random.choice(indices, k, replace=False)
    }
    
    # Compute similarities for each selection
    similarities = {}
    for selection_name, selection_indices in selections.items():
        similarities[selection_name] = compute_similarities_for_subset(
            matching_activations,
            non_matching_activations,
            selection_indices,
            device
        )
    
    return similarities

def plot_cosine_distributions(
    similarities_dict: Dict[str, List[float]],
    title: str,
    ax: plt.Axes
) -> None:
    """Plot histograms of cosine similarities."""
    colors = {
        'all': 'grey',
        'top': 'green',
        'middle': 'yellow',
        'bottom': 'red',
        'random': 'violet'
    }
    
    all_sims = [s for sims in similarities_dict.values() for s in sims]
    min_sim, max_sim = min(all_sims), max(all_sims)
    bins = np.linspace(min_sim, max_sim, 30)
    
    # Plot in specific order
    for subset_name in colors.keys():
        similarities = similarities_dict[subset_name]
        alpha = 0.3 if subset_name == 'all' else 0.7
        ax.hist(similarities, bins=bins, alpha=alpha,
                label=subset_name, color=colors[subset_name])
    
    ax.set_title(title)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.legend()

def analyze_prompt_type(
    dataset_name: str,
    prompt_type: str,
    device: str
) -> Dict[str, List[float]]:
    """Analyze cosine similarities for a specific prompt type."""
    # Load paired activations and logits
    paired_data = utils.load_paired_activations_and_logits(
        dataset_name=dataset_name,
        pairing_type=prompt_type,
        num_samples=NUM_SAMPLES
    )
    
    # Get probability differences using logits
    matching_logits = torch.tensor(paired_data['matching_logits'])
    non_matching_logits = torch.tensor(paired_data['non_matching_logits'])
    probability_differences = utils.compute_probability_differences(
        matching_logits,
        non_matching_logits,
        paired_data['metadata']['matching_answers']
    )
    
    # Compute similarities
    similarities = select_samples_and_compute_similarities(
        paired_data['matching_activations'],
        paired_data['non_matching_activations'],
        probability_differences,
        device
    )
    
    return similarities

def main() -> None:
    device = utils.get_device()
    prompt_types = ['prefilled_answer', 'instruction', '5-shot', 'prefilled_instruction', 'prefilled_5-shot', 'instruction_5-shot', 'prefilled_instruction_5-shot']

    datasets = all_datasets_with_type_figure_13[:NUM_DATASETS]
    
    # Process each prompt type
    for prompt_type in prompt_types:
        logger.info(f"Processing prompt type: {prompt_type}")
        
        # Create figure for this prompt type
        n_rows = (NUM_DATASETS + 1 + 2) // 3  # +1 for average plot, +2 for ceiling division
        fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axs = axs.flatten()
        
        # Store similarities for computing average
        all_dataset_similarities = []
        
        # Process each dataset
        for idx, (dataset_name, _) in enumerate(tqdm(datasets)):
            similarities = analyze_prompt_type(dataset_name, prompt_type, device)
            all_dataset_similarities.append(similarities)
            
            # Plot individual dataset results
            plot_cosine_distributions(
                similarities,
                f"{dataset_name} - {prompt_type}",
                axs[idx+1]  # +1 because first plot will be average
            )
        
        # Compute and plot average across datasets
        avg_similarities = {key: [] for key in ['all', 'top', 'middle', 'bottom', 'random']}
        for similarities in all_dataset_similarities:
            for key in avg_similarities:
                avg_similarities[key].extend(similarities[key])
                
        plot_cosine_distributions(
            avg_similarities,
            f"Average Across Datasets - {prompt_type}",
            axs[0]
        )
        
        # Remove empty subplots
        for idx in range(len(datasets) + 1, len(axs)):
            fig.delaxes(axs[idx])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        DATA_PATH = utils.get_path('DATA_PATH')
        output_dir = os.path.join(DATA_PATH, 'cosine_similarity_results')
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(
            os.path.join(
                output_dir,
                f'cosine_similarities_{prompt_type}_{timestamp}.pdf'
            )
        )
        plt.close()

if __name__ == "__main__":
    main()