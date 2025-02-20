import os
import pickle
import random
from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
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

def directional_magnitude(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute how far v1 goes in direction of v2."""
    v2_norm = torch.norm(v2.flatten())
    projection = torch.dot(v1.flatten(), v2.flatten()) / v2_norm
    return float(projection / v2_norm)  # Normalize by v2's magnitude

def load_stored_activations(activations_path: str, dataset_name: str, 
                          model_name: str, num_samples: int, scenario: str) -> List:
    file_path = os.path.join(
        activations_path,
        dataset_name,
        f"{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    with open(file_path, 'rb') as f:
        return pickle.load(f)['results'][scenario]

def plot_magnitude_distributions(magnitudes_dict: Dict[str, List[float]], 
                               dataset_name: str, ax: plt.Axes, k: int) -> None:
    """Plot histograms of directional magnitudes."""
    colors = {
        'all': 'grey',
        'top': 'green',
        'middle': 'yellow',
        'bottom': 'red',
        'random': 'violet'
    }
    
    all_mags = [m for mags in magnitudes_dict.values() for m in mags]
    min_mag, max_mag = min(all_mags), max(all_mags)
    bins = np.linspace(min_mag, max_mag, 30)
    
    for subset_name in colors.keys():
        magnitudes = magnitudes_dict[subset_name]
        alpha = 0.3 if subset_name == 'all' else 0.7
        ax.hist(magnitudes, bins=bins, alpha=alpha,
                label=subset_name, color=colors[subset_name])
    
    ax.set_title(f"{dataset_name} (K={k})")
    ax.set_xlabel('Magnitude in Steering Direction')
    ax.set_ylabel('Count')
    ax.legend()

def compute_magnitudes_for_dataset(pos_data: List, neg_data: List, 
                                 device: str) -> Dict[str, List[float]]:
    """Compute directional magnitudes for all subsets of a dataset."""
    full_pos_activations = [sample['activation_layer_13'] for sample in pos_data]
    full_neg_activations = [sample['activation_layer_13'] for sample in neg_data]
    full_steering_vector = utils.compute_contrastive_steering_vector(
        full_pos_activations, full_neg_activations, device
    )
    
    magnitudes = []
    for pos, neg in zip(pos_data, neg_data):
        indiv_vector = compute_individual_steering_vector(
            pos['activation_layer_13'], 
            neg['activation_layer_13'],
            device
        )
        mag = directional_magnitude(indiv_vector, full_steering_vector)
        magnitudes.append(mag)
    
    sorted_indices = np.argsort(magnitudes)
    n = len(magnitudes)
    k = K_SAMPLES
    
    return {
        'top': [magnitudes[i] for i in sorted_indices[-k:]],
        'middle': [magnitudes[i] for i in sorted_indices[n//2-k//2:n//2+k//2]],
        'bottom': [magnitudes[i] for i in sorted_indices[:k]],
        'random': [magnitudes[i] for i in np.random.choice(sorted_indices, k, replace=False)],
        'all': magnitudes
    }

def main():
    MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
    
    device = utils.get_device()
    torch.set_grad_enabled(False)
    activations_path = os.path.join(utils.get_path('DISK_PATH'), "reliability_paper_activations")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATASETS]
    all_magnitudes = []
    
    n_rows = (NUM_DATASETS + 4) // 3
    fig, axs = plt.subplots(n_rows, 3, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for idx, (dataset_name, _) in enumerate(tqdm(datasets, desc="Processing datasets")):
        pos_data = load_stored_activations(activations_path, dataset_name, 
                                         'llama2_7b_chat', TRAIN_SAMPLES, 
                                         'matching_instruction')
        neg_data = load_stored_activations(activations_path, dataset_name, 
                                         'llama2_7b_chat', TRAIN_SAMPLES, 
                                         'non_matching_instruction')
        
        magnitudes_dict = compute_magnitudes_for_dataset(pos_data, neg_data, device)
        all_magnitudes.append(magnitudes_dict)
        
        plot_magnitude_distributions(magnitudes_dict, dataset_name, axs[idx+1], K_SAMPLES)
    
    avg_magnitudes = {key: [] for key in ['top', 'middle', 'bottom', 'random', 'all']}
    for magnitudes_dict in all_magnitudes:
        for key in avg_magnitudes:
            avg_magnitudes[key].extend(magnitudes_dict[key])
            
    plot_magnitude_distributions(avg_magnitudes, "Average Across Datasets", axs[0], K_SAMPLES)
    
    for idx in range(len(datasets) + 1, len(axs)):
        fig.delaxes(axs[idx])
    
    plt.tight_layout()
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'steering_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'directional_magnitudes_distributions.pdf'))
    plt.close()

if __name__ == "__main__":
    main()