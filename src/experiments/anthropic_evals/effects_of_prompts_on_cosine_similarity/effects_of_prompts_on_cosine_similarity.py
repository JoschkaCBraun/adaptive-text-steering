import os
import logging
from typing import Dict, List
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13

# Constants
NUM_SAMPLES = 200
NUM_DATASETS = 6
PROMPT_TYPES = [
    'prefilled_answer',
    'instruction',
    '5-shot',
    'prefilled_instruction',
    'prefilled_5-shot',
    'instruction_5-shot',
    'prefilled_instruction_5-shot'
]

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def compute_similarities(
    matching_activations: np.ndarray,
    non_matching_activations: np.ndarray,
    device: str
) -> List[float]:
    """Compute cosine similarities between difference vectors and steering vector."""
    # Convert activations to float32 tensors
    matching_tensors = [torch.tensor(row, dtype=torch.float32) for row in matching_activations]
    non_matching_tensors = [torch.tensor(row, dtype=torch.float32) for row in non_matching_activations]
    
    # Move tensors to device
    matching_tensors = [t.to(device) for t in matching_tensors]
    non_matching_tensors = [t.to(device) for t in non_matching_tensors]
    
    # Compute steering vector
    steering_vector = utils.compute_contrastive_steering_vector(
        matching_tensors,
        non_matching_tensors,
        device
    )
    
    # Compute difference vectors
    difference_vectors = [m - n for m, n in zip(matching_tensors, non_matching_tensors)]
    
    # Compute similarities
    similarities = utils.compute_many_against_one_cosine_similarity(
        difference_vectors,
        steering_vector,
        device
    )
    
    return similarities

def analyze_dataset(
    dataset_name: str,
    device: str
) -> Dict[str, List[float]]:
    """Analyze cosine similarities for all prompt types for a specific dataset."""
    logger.info(f"Analyzing dataset: {dataset_name}")
    
    similarities_by_prompt = {}
    for prompt_type in PROMPT_TYPES:
        try:
            # Load paired data
            paired_data = utils.load_paired_activations_and_logits(
                dataset_name=dataset_name,
                pairing_type=prompt_type,
                num_samples=NUM_SAMPLES
            )
            
            # Ensure arrays are float32
            matching_activations = paired_data['matching_activations'].astype(np.float32)
            non_matching_activations = paired_data['non_matching_activations'].astype(np.float32)
            
            # Compute similarities
            similarities = compute_similarities(
                matching_activations,
                non_matching_activations,
                device
            )
            
            similarities_by_prompt[prompt_type] = similarities
            
        except Exception as e:
            logger.error(f"Error processing {prompt_type} for {dataset_name}: {str(e)}")
            similarities_by_prompt[prompt_type] = []
    
    return similarities_by_prompt

def plot_similarity_distributions(
    all_similarities: Dict[str, Dict[str, List[float]]],
    output_dir: str
) -> None:
    """Plot similarity distributions for all datasets and prompt types."""
    # Get a color map with enough distinct colors
    n_colors = len(PROMPT_TYPES)
    color_map = plt.colormaps.get_cmap('tab10')
    colors = [color_map(i/n_colors) for i in range(n_colors)]
    
    # Create vertical subplots with an extra row for the average
    total_plots = NUM_DATASETS + 1  # Add 1 for average plot
    fig, axs = plt.subplots(total_plots, 1, figsize=(7, 3*total_plots))
    if total_plots == 1:
        axs = [axs]
    
    # Calculate and plot average distributions first
    ax_avg = axs[0]
    avg_similarities_by_prompt = {}
    
    # Combine similarities across datasets
    for prompt_type in PROMPT_TYPES:
        all_sims = []
        for dataset_sims in all_similarities.values():
            if prompt_type in dataset_sims and dataset_sims[prompt_type]:
                all_sims.extend(dataset_sims[prompt_type])
        if all_sims:
            avg_similarities_by_prompt[prompt_type] = all_sims
    
    # Plot average distributions
    if avg_similarities_by_prompt:
        all_avg_sims = [s for sims in avg_similarities_by_prompt.values() for s in sims]
        min_sim, max_sim = min(all_avg_sims), max(all_avg_sims)
        bins = np.linspace(min_sim, max_sim, 30)
        
        for prompt_idx, (prompt_type, similarities) in enumerate(avg_similarities_by_prompt.items()):
            color = colors[PROMPT_TYPES.index(prompt_type)]
            mean_sim = np.mean(similarities)
            
            ax_avg.hist(similarities, bins=bins, alpha=0.5,
                       label=f'{prompt_type} \n(mean={mean_sim:.3f})', 
                       color=color)
            ax_avg.axvline(x=mean_sim, color=color, linestyle='--', alpha=0.8)
        
        ax_avg.set_title(f'Average Across All Datasets (n={NUM_SAMPLES})', pad=10)
        ax_avg.set_xlabel('Cosine Similarity to Steering Vector')
        ax_avg.set_ylabel('Count')
        ax_avg.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
                     prop={'size': 8})  # Smaller legend font
    
    # Plot individual dataset distributions
    for idx, (dataset_name, similarities_by_prompt) in enumerate(all_similarities.items()):
        ax = axs[idx + 1]  # +1 to account for average plot
        
        # Filter out empty similarity lists
        similarities_by_prompt = {k: v for k, v in similarities_by_prompt.items() if v}
        
        if similarities_by_prompt:  # Only plot if we have data
            all_sims = [s for sims in similarities_by_prompt.values() for s in sims]
            min_sim, max_sim = min(all_sims), max(all_sims)
            bins = np.linspace(min_sim, max_sim, 30)
            
            for prompt_idx, (prompt_type, similarities) in enumerate(similarities_by_prompt.items()):
                color = colors[PROMPT_TYPES.index(prompt_type)]
                mean_sim = np.mean(similarities)
                
                ax.hist(similarities, bins=bins, alpha=0.5,
                       label=f'{prompt_type} \n(mean={mean_sim:.3f})', 
                       color=color)
                ax.axvline(x=mean_sim, color=color, linestyle='--', alpha=0.8)
            
            ax.set_title(f'{dataset_name} (n={NUM_SAMPLES})', pad=10)
            ax.set_xlabel('Cosine Similarity to Steering Vector')
            ax.set_ylabel('Count')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                     prop={'size': 8})  # Smaller legend font
    
    # Adjust layout and save
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(
            output_dir,
            f'cosine_similarity_distributions_{timestamp}.pdf'
        ),
        bbox_inches='tight'
    )
    plt.close()

def main() -> None:
    # Setup
    device = utils.get_device()
    datasets = all_datasets_with_type_figure_13[:NUM_DATASETS]
    
    # Setup output directory
    DATA_PATH = utils.get_path('DATA_PATH')
    output_dir = os.path.join(DATA_PATH, 'anthropic_evals_results', 'prompt_cosine_similarity_distributions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all datasets
    all_similarities = {}
    for dataset_name, _ in datasets:
        similarities = analyze_dataset(dataset_name, device)
        all_similarities[dataset_name] = similarities
    
    # Generate plots
    plot_similarity_distributions(all_similarities, output_dir)

if __name__ == "__main__":
    main()