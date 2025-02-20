import os
import pickle
import random
from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.dataset_names import all_datasets_with_type_figure
import src.utils as utils

TRAIN_SAMPLES = 200
NUM_DATASETS = 36
K_SAMPLES = 40

def compute_individual_steering_vector(pos_activation: torch.Tensor, 
                                    neg_activation: torch.Tensor,
                                    device: str) -> torch.Tensor:
    """Compute steering vector for a single sample pair."""
    pos = torch.mean(torch.stack([pos_activation]), dim=0)
    neg = torch.mean(torch.stack([neg_activation]), dim=0)
    return (pos - neg).to(device)

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return float(torch.nn.functional.cosine_similarity(v1.flatten(), v2.flatten(), dim=0))

def load_stored_activations(activations_path: str, dataset_name: str, 
                          model_name: str, num_samples: int, scenario: str) -> List:
    file_path = os.path.join(
        activations_path,
        dataset_name,
        f"{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    with open(file_path, 'rb') as f:
        return pickle.load(f)['results'][scenario]

def plot_cosine_distributions(similarities_dict: Dict[str, List[float]], 
                            dataset_name: str, ax: plt.Axes, k: int) -> None:
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
    
    # Plot in specific order: grey first, then others
    for subset_name in colors.keys():
        similarities = similarities_dict[subset_name]
        alpha = 0.3 if subset_name == 'all' else 0.7
        ax.hist(similarities, bins=bins, alpha=alpha,
                label=subset_name, color=colors[subset_name])
    
    ax.set_title(f"{dataset_name} (K={k})")
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.legend()

def compute_similarities_for_dataset(pos_data: List, neg_data: List, 
                                  device: str) -> Dict[str, List[float]]:
    """Compute cosine similarities for all subsets of a dataset."""
    # Compute full steering vector
    full_pos_activations = [sample['activation_layer_13'] for sample in pos_data]
    full_neg_activations = [sample['activation_layer_13'] for sample in neg_data]
    full_steering_vector = utils.compute_contrastive_steering_vector(
        full_pos_activations, full_neg_activations, device
    )
    
    # Compute individual similarities
    similarities = []
    for pos, neg in zip(pos_data, neg_data):
        indiv_vector = compute_individual_steering_vector(
            pos['activation_layer_13'], 
            neg['activation_layer_13'],
            device
        )
        sim = cosine_similarity(indiv_vector, full_steering_vector)
        similarities.append(sim)
    
    # Sort similarities for different subsets
    sorted_indices = np.argsort(similarities)
    n = len(similarities)
    k = K_SAMPLES
    
    return {
        'top': [similarities[i] for i in sorted_indices[-k:]],
        'middle': [similarities[i] for i in sorted_indices[n//2-k//2:n//2+k//2]],
        'bottom': [similarities[i] for i in sorted_indices[:k]],
        'random': [similarities[i] for i in np.random.choice(sorted_indices, k, replace=False)],
        'all': similarities
    }

def main():
    # Configuration

    MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
    
    # Setup
    device = utils.get_device()
    torch.set_grad_enabled(False)
    activations_path = os.path.join(utils.get_path('DISK_PATH'), "reliability_paper_activations")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Process datasets
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATASETS]
    all_similarities = []
    
    # Create figure for subplots
    n_rows = (NUM_DATASETS + 4) // 3  # Ensures enough rows for all plots
    fig, axs = plt.subplots(n_rows, 3, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    # Process each dataset
    for idx, (dataset_name, _) in enumerate(tqdm(datasets, desc="Processing datasets")):
        # Load data
        pos_data = load_stored_activations(activations_path, dataset_name, 
                                         'llama2_7b_chat', TRAIN_SAMPLES, 
                                         'matching_instruction')
        neg_data = load_stored_activations(activations_path, dataset_name, 
                                         'llama2_7b_chat', TRAIN_SAMPLES, 
                                         'non_matching_instruction')
        
        # Compute similarities
        similarities_dict = compute_similarities_for_dataset(pos_data, neg_data, device)
        all_similarities.append(similarities_dict)
        
        # Plot individual dataset
        plot_cosine_distributions(similarities_dict, dataset_name, axs[idx+1], K_SAMPLES)
    
    # Compute and plot average across datasets
    avg_similarities = {key: [] for key in ['top', 'middle', 'bottom', 'random', 'all']}
    for similarities_dict in all_similarities:
        for key in avg_similarities:
            avg_similarities[key].extend(similarities_dict[key])
            
    plot_cosine_distributions(avg_similarities, "Average Across Datasets", axs[0], K_SAMPLES)
    
    # Remove empty subplots
    for idx in range(len(datasets) + 1, len(axs)):
        fig.delaxes(axs[idx])
    
    # Save plot
    plt.tight_layout()
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'steering_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cosine_similarities_distributions.pdf'))
    plt.close()

if __name__ == "__main__":
    main()