"""
analyse_effect_of_prompts_on_projection_magnitude_compare_subsets.py
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

def compute_projections_for_subset(
    matching_activations: torch.Tensor,
    non_matching_activations: torch.Tensor,
    indices: np.ndarray,
    device: str
) -> List[float]:
    """Compute projection magnitudes along steering vector direction for a subset of samples."""
    # Get activations for the subset and convert to list of tensors on the correct device
    matching_subset = [torch.tensor(row, device=device) for row in matching_activations[indices]]
    non_matching_subset = [torch.tensor(row, device=device) for row in non_matching_activations[indices]]
    
    # Compute steering vector for this subset
    steering_vector = utils.compute_contrastive_steering_vector(
        matching_subset, 
        non_matching_subset,
        device
    )
    
    # Normalize the steering vector
    steering_vector_norm = torch.norm(steering_vector)
    normalized_steering_vector = steering_vector / steering_vector_norm
    
    # Prepare difference vectors (already on the correct device from previous step)
    difference_vectors = [m - n for m, n in zip(matching_subset, non_matching_subset)]
    
    # Compute projections onto the normalized steering vector
    projections = []
    for diff_vec in difference_vectors:
        # Dot product with normalized steering vector gives us the magnitude along that direction
        projection = torch.dot(diff_vec, normalized_steering_vector)
        projections.append(projection.item())
    
    return projections

def select_samples_and_compute_projections(
    matching_activations: np.ndarray,
    non_matching_activations: np.ndarray,
    probability_differences: List[float],
    device: str,
    k: int = K_SAMPLES
) -> Dict[str, List[float]]:
    """Select different subsets of samples and compute their projection magnitudes."""
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
    
    # Compute projections for each selection
    projections = {}
    for selection_name, selection_indices in selections.items():
        projections[selection_name] = compute_projections_for_subset(
            matching_activations,
            non_matching_activations,
            selection_indices,
            device
        )
    
    return projections

def plot_projection_distributions(
    projections_dict: Dict[str, List[float]],
    title: str,
    ax: plt.Axes
) -> None:
    """Plot histograms of projection magnitudes."""
    colors = {
        'all': 'grey',
        'top': 'green',
        'middle': 'yellow',
        'bottom': 'red',
        'random': 'violet'
    }
    
    all_projs = [p for projs in projections_dict.values() for p in projs]
    min_proj, max_proj = min(all_projs), max(all_projs)
    bins = np.linspace(min_proj, max_proj, 30)
    
    # Plot in specific order
    for subset_name in colors.keys():
        projections = projections_dict[subset_name]
        alpha = 0.3 if subset_name == 'all' else 0.7
        ax.hist(projections, bins=bins, alpha=alpha,
                label=subset_name, color=colors[subset_name])
    
    ax.set_title(title)
    ax.set_xlabel('Projection Magnitude')
    ax.set_ylabel('Count')
    ax.legend()

def analyze_prompt_type(
    dataset_name: str,
    prompt_type: str,
    device: str
) -> Dict[str, List[float]]:
    """Analyze projection magnitudes for a specific prompt type."""
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
    
    # Compute projections
    projections = select_samples_and_compute_projections(
        paired_data['matching_activations'],
        paired_data['non_matching_activations'],
        probability_differences,
        device
    )
    
    return projections

def main() -> None:
    device = utils.get_device()
    prompt_types = ['prefilled_answer', 'instruction', '5-shot', 'prefilled_instruction', 
                   'prefilled_5-shot', 'instruction_5-shot', 'prefilled_instruction_5-shot']

    datasets = all_datasets_with_type_figure_13[:NUM_DATASETS]
    
    # Process each prompt type
    for prompt_type in prompt_types:
        logger.info(f"Processing prompt type: {prompt_type}")
        
        # Create figure for this prompt type
        n_rows = (NUM_DATASETS + 1 + 2) // 3  # +1 for average plot, +2 for ceiling division
        fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axs = axs.flatten()
        
        # Store projections for computing average
        all_dataset_projections = []
        
        # Process each dataset
        for idx, (dataset_name, _) in enumerate(tqdm(datasets)):
            projections = analyze_prompt_type(dataset_name, prompt_type, device)
            all_dataset_projections.append(projections)
            
            # Plot individual dataset results
            plot_projection_distributions(
                projections,
                f"{dataset_name} - {prompt_type}",
                axs[idx+1]  # +1 because first plot will be average
            )
        
        # Compute and plot average across datasets
        avg_projections = {key: [] for key in ['all', 'top', 'middle', 'bottom', 'random']}
        for projections in all_dataset_projections:
            for key in avg_projections:
                avg_projections[key].extend(projections[key])
                
        plot_projection_distributions(
            avg_projections,
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
        output_dir = os.path.join('data', 'projection_magnitude_results')
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(
            os.path.join(
                output_dir,
                f'projection_magnitudes_{prompt_type}_{timestamp}.pdf'
            )
        )
        plt.close()

if __name__ == "__main__":
    main()