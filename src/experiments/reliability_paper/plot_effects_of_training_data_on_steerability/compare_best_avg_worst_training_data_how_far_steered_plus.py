import os
import pickle
import random
from typing import Dict, List, Tuple
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
    pos = torch.mean(torch.stack([pos_activation]), dim=0)
    neg = torch.mean(torch.stack([neg_activation]), dim=0)
    return (pos - neg).to(device)

def compute_metrics(v1: torch.Tensor, v2: torch.Tensor) -> Tuple[float, float]:
    """Compute directional magnitude and normalized L1 norm."""
    v2_norm = torch.norm(v2.flatten())
    # Directional magnitude
    projection = torch.dot(v1.flatten(), v2.flatten()) / v2_norm
    dir_magnitude = float(projection / v2_norm)
    # Normalized L1 norm
    norm = float(torch.norm(v1.flatten(), p=1) / v2_norm)
    return dir_magnitude, norm

def load_stored_activations(activations_path: str, dataset_name: str, 
                          model_name: str, num_samples: int, scenario: str) -> List:
    file_path = os.path.join(
        activations_path,
        dataset_name,
        f"{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    with open(file_path, 'rb') as f:
        return pickle.load(f)['results'][scenario]


def plot_distributions(data_dict: Dict[str, List[float]], 
                      dataset_name: str, ax: plt.Axes, k: int,
                      metric_name: str) -> None:
    colors = {
        'all': 'grey',
        'top': 'green',
        'middle': 'yellow',
        'bottom': 'red',
        'random': 'violet'
    }
    
    all_vals = [v for vals in data_dict.values() for v in vals]
    min_val, max_val = min(all_vals), max(all_vals)
    bins = np.linspace(min_val, max_val, 30)
    
    for subset_name in colors.keys():
        values = data_dict[subset_name]
        alpha = 0.3 if subset_name == 'all' else 0.7
        ax.hist(values, bins=bins, alpha=alpha,
               label=subset_name, color=colors[subset_name])
    
    ax.set_title(f"{dataset_name} (K={k})")
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Count')
    ax.legend()

def compute_metrics_for_dataset(pos_data: List, neg_data: List, 
                              device: str) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    full_pos_activations = [sample['activation_layer_13'] for sample in pos_data]
    full_neg_activations = [sample['activation_layer_13'] for sample in neg_data]
    full_steering_vector = utils.compute_contrastive_steering_vector(
        full_pos_activations, full_neg_activations, device
    )
    
    magnitudes = []
    norms = []
    for pos, neg in zip(pos_data, neg_data):
        indiv_vector = compute_individual_steering_vector(
            pos['activation_layer_13'], 
            neg['activation_layer_13'],
            device
        )
        mag, norm = compute_metrics(indiv_vector, full_steering_vector)
        magnitudes.append(mag)
        norms.append(norm)
    
    sorted_mag_indices = np.argsort(magnitudes)
    sorted_norm_indices = np.argsort(norms)
    n = len(magnitudes)
    k = K_SAMPLES
    
    mag_dict = {
        'top': [magnitudes[i] for i in sorted_mag_indices[-k:]],
        'middle': [magnitudes[i] for i in sorted_mag_indices[n//2-k//2:n//2+k//2]],
        'bottom': [magnitudes[i] for i in sorted_mag_indices[:k]],
        'random': [magnitudes[i] for i in np.random.choice(sorted_mag_indices, k, replace=False)],
        'all': magnitudes
    }
    
    norm_dict = {
        'top': [norms[i] for i in sorted_norm_indices[-k:]],
        'middle': [norms[i] for i in sorted_norm_indices[n//2-k//2:n//2+k//2]],
        'bottom': [norms[i] for i in sorted_norm_indices[:k]],
        'random': [norms[i] for i in np.random.choice(sorted_norm_indices, k, replace=False)],
        'all': norms
    }
    
    return mag_dict, norm_dict

def create_plots(all_magnitudes: List[Dict], all_norms: List[Dict], 
                datasets: List, output_dir: str) -> None:
    n_rows = (NUM_DATASETS + 4) // 3
    
    # Plot magnitudes
    fig_mag, axs_mag = plt.subplots(n_rows, 3, figsize=(12, 3*n_rows))
    axs_mag = axs_mag.flatten()
    
    # Plot average magnitudes
    avg_magnitudes = {key: [] for key in ['top', 'middle', 'bottom', 'random', 'all']}
    for magnitudes_dict in all_magnitudes:
        for key in avg_magnitudes:
            avg_magnitudes[key].extend(magnitudes_dict[key])
    plot_distributions(avg_magnitudes, "Average Across Datasets", 
                      axs_mag[0], K_SAMPLES, 'Magnitude in Steering Direction')
    
    # Plot individual dataset magnitudes
    for idx, (dataset_name, _) in enumerate(datasets):
        plot_distributions(all_magnitudes[idx], dataset_name, 
                         axs_mag[idx+1], K_SAMPLES, 'Magnitude in Steering Direction')
    
    # Remove empty subplots for magnitudes
    for idx in range(len(datasets) + 1, len(axs_mag)):
        fig_mag.delaxes(axs_mag[idx])
    
    plt.figure(fig_mag.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'directional_magnitudes_distributions.pdf'))
    plt.close()
    
    # Plot norms
    fig_norm, axs_norm = plt.subplots(n_rows, 3, figsize=(12, 3*n_rows))
    axs_norm = axs_norm.flatten()
    
    # Plot average norms
    avg_norms = {key: [] for key in ['top', 'middle', 'bottom', 'random', 'all']}
    for norms_dict in all_norms:
        for key in avg_norms:
            avg_norms[key].extend(norms_dict[key])
    plot_distributions(avg_norms, "Average Across Datasets", 
                      axs_norm[0], K_SAMPLES, 'Normalized L1 Norm')
    
    # Plot individual dataset norms
    for idx, (dataset_name, _) in enumerate(datasets):
        plot_distributions(all_norms[idx], dataset_name, 
                         axs_norm[idx+1], K_SAMPLES, 'Normalized L1 Norm')
    
    # Remove empty subplots for norms
    for idx in range(len(datasets) + 1, len(axs_norm)):
        fig_norm.delaxes(axs_norm[idx])
    
    plt.figure(fig_norm.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_l1_norms_distributions.pdf'))
    plt.close()

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
    all_norms = []
    
    for dataset_name, _ in tqdm(datasets, desc="Processing datasets"):
        pos_data = load_stored_activations(activations_path, dataset_name, 
                                         'llama2_7b_chat', TRAIN_SAMPLES, 
                                         'matching_instruction')
        neg_data = load_stored_activations(activations_path, dataset_name, 
                                         'llama2_7b_chat', TRAIN_SAMPLES, 
                                         'non_matching_instruction')
        
        mag_dict, norm_dict = compute_metrics_for_dataset(pos_data, neg_data, device)
        all_magnitudes.append(mag_dict)
        all_norms.append(norm_dict)
    
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'steering_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    create_plots(all_magnitudes, all_norms, datasets, output_dir)

if __name__ == "__main__":
    main()