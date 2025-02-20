"""
Plot cosine similarities between different instruction scenarios and their steering vectors.
"""
import os
import pickle
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_contrastive_activations(positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> torch.Tensor:
    """Compute contrastive activations between positive and negative samples."""
    return positive_acts - negative_acts

def compute_steering_vector(positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> torch.Tensor:
    """Compute steering vector as mean difference between positive and negative activations."""
    return torch.mean(positive_acts - negative_acts, dim=0)

def compute_cosine_similarities(activations: torch.Tensor) -> List[float]:
    """Compute pairwise cosine similarities between activations."""
    n_samples = activations.shape[0]
    similarities = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sim = torch.nn.functional.cosine_similarity(
                activations[i].unsqueeze(0), 
                activations[j].unsqueeze(0)
            ).item()
            similarities.append(sim)
    
    return similarities

def compute_steering_similarities(activations: torch.Tensor, steering_vector: torch.Tensor) -> List[float]:
    """Compute cosine similarities between activations and steering vector."""
    similarities = []
    
    for activation in activations:
        sim = torch.nn.functional.cosine_similarity(
            activation.unsqueeze(0),
            steering_vector.unsqueeze(0)
        ).item()
        similarities.append(sim)
    
    return similarities

def process_scenario_pair(
    data: Dict,
    positive_scenario: str,
    negative_scenario: str
) -> Tuple[List[float], List[float]]:
    """Process a pair of scenarios to get cosine similarities."""
    # Extract activations
    positive_acts = torch.stack([
        torch.as_tensor(sample['activation_layer_13']).clone().detach()
        for sample in data['results'][positive_scenario]
    ])
    negative_acts = torch.stack([
        torch.as_tensor(sample['activation_layer_13']).clone().detach()
        for sample in data['results'][negative_scenario]
    ])
    
    # Compute contrastive activations and steering vector
    contrastive_acts = compute_contrastive_activations(positive_acts, negative_acts)
    steering_vector = compute_steering_vector(positive_acts, negative_acts)
    
    # Compute similarities
    pairwise_sims = compute_cosine_similarities(contrastive_acts)
    steering_sims = compute_steering_similarities(contrastive_acts, steering_vector)
    
    return pairwise_sims, steering_sims

def plot_distributions(
    data_dict: Dict,
    dataset_name: str,
    ax1: plt.Axes,
    ax2: plt.Axes
) -> None:
    """Plot cosine similarity distributions for a dataset."""
    # Define colors and scenarios
    scenarios = [
        ('matching_prefilled', 'non_matching_prefilled', 'Prefilled', '#1f77b4'),
        ('matching_instruction', 'non_matching_instruction', 'Regular Instructions', '#ff7f0e'),
        ('matching_instruction_prefilled', 'non_matching_instruction_prefilled', 'Prefilled Instructions', '#d62728'),
        ('matching_instruction_few_shot', 'non_matching_instruction_few_shot', 'Few-shot Instructions', '#2ca02c'),
        ('matching_instruction_few_shot_prefilled', 'non_matching_instruction_few_shot_prefilled', 'Few-shot Prefilled Instructions', '#9467bd')
    ]
    
    # Process each scenario pair
    stats_text = []
    
    for pos_scenario, neg_scenario, label, color in scenarios:
        pair_sims, steer_sims = process_scenario_pair(
            data_dict,
            pos_scenario,
            neg_scenario
        )
        
        # Plot distributions
        ax1.hist(pair_sims, bins=20, density=True, alpha=0.6, color=color,
                 label=label, histtype='step', linewidth=2)
        ax2.hist(steer_sims, bins=20, density=True, alpha=0.6, color=color,
                 label=label, histtype='step', linewidth=2)
        
        # Collect statistics
        stats_text.append(
            f'{label}:\n'
            f'  Pair Mean: {np.mean(pair_sims):.3f}\n'
            # f'  Pair Std: {np.std(pair_sims):.3f}\n'
            f'  Steer Mean: {np.mean(steer_sims):.3f}\n'
            # f'  Steer Std: {np.std(steer_sims):.3f}\n'
        )
    
    # Add statistics text to plots
    for ax, title in [
        (ax1, 'Pairwise Cosine Similarities'),
        (ax2, 'Steering Vector Cosine Similarities')
    ]:
        ax.text(0.02, 0.98, ''.join(stats_text), transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title(f'{title}\n{dataset_name}')
        ax.legend()
        ax.set_xlim(-1, 1)

def main() -> None:
    # Configuration
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    DATA_PATH = utils.get_path('DATA_PATH')
    PLOTS_PATH = os.path.join(DATA_PATH, "reliability_paper_data", "plots")
    model_name = "llama2_7b_chat"
    NUM_SAMPLES = 40
    
    datasets = all_datasets_with_type_figure_13_tan_et_al[::5]
    
    # Create figure with subplots for each dataset
    fig, axs = plt.subplots(len(datasets), 2, figsize=(15, 5 * len(datasets)))
    fig.suptitle('Cosine Similarity Distributions Across Datasets', fontsize=16, y=0.95)
    
    # Process each dataset
    for idx, (dataset_name, dataset_type) in enumerate(datasets):
        logging.info(f"Processing dataset: {dataset_name}")
        
        # Load data
        file_path = os.path.join(
            ACTIVATIONS_PATH,
            f"{dataset_name}/{model_name}_{dataset_name}_activations_and_logits_for_{NUM_SAMPLES}_samples.pkl"
        )
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Get current row of subplots
            ax1 = axs[idx, 0] if len(datasets) > 1 else axs[0]
            ax2 = axs[idx, 1] if len(datasets) > 1 else axs[1]
            
            # Plot distributions
            plot_distributions(data, dataset_name, ax1, ax2)
            
        except Exception as e:
            logging.error(f"Error processing {dataset_name}: {str(e)}")
            continue
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(PLOTS_PATH, f"cosine_similarity_distributions_for_{NUM_SAMPLES}_samples.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()