"""
plot_kappa_distribution_for_steered_and_non_steered_activations.py
"""

import os
import pickle
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from scipy.special import kl_div
import torch

from src.utils.dataset_names import all_datasets_with_index_and_fraction_anti_steerable_figure_13
import src.utils.compute_steering_vectors as sv
import src.utils as utils


def compute_separation_metrics(pos_proj: np.ndarray, neg_proj: np.ndarray) -> dict:
    """Compute statistical separation metrics between positive and negative projections."""
    pos_mean, neg_mean = np.mean(pos_proj), np.mean(neg_proj)
    pos_std, neg_std = np.std(pos_proj), np.std(neg_proj)
    
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    cohens_d = (pos_mean - neg_mean) / pooled_std
    
    bins = np.linspace(min(np.min(pos_proj), np.min(neg_proj)),
                       max(np.max(pos_proj), np.max(neg_proj)), 50)
    pos_hist, _ = np.histogram(pos_proj, bins=bins, density=True)
    neg_hist, _ = np.histogram(neg_proj, bins=bins, density=True)
    overlap_coef = np.sum(np.minimum(pos_hist, neg_hist)) * (bins[1] - bins[0])
    
    eps = 1e-10
    pos_hist = (pos_hist + eps) / np.sum(pos_hist + eps)
    neg_hist = (neg_hist + eps) / np.sum(neg_hist + eps)
    kl = np.sum(kl_div(pos_hist, neg_hist))
    
    return {
        'pos_mean': pos_mean, 'neg_mean': neg_mean,
        'pos_std': pos_std, 'neg_std': neg_std,
        'cohens_d': cohens_d, 'overlap_coef': overlap_coef,
        'kl_divergence': kl
    }


def load_activations(file_path: str, num_samples: int, device: torch.device) -> dict:
    """Load and prepare activations for kappa calculation."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    result = {
        'positive': {},
        'negative': {}
    }
    
    for class_type in ['positive', 'negative']:
        for layer in data[class_type].keys():
            layer_tensors = [tensor.to(device) for tensor in data[class_type][layer][:num_samples]]
            result[class_type][int(layer)] = layer_tensors
            
    return result


def apply_steering_vector(activations: dict, steering_vector: torch.Tensor, 
                          layer: int, factor: float) -> dict:
    """Apply steering vector to activations with given factor."""
    result = {
        'positive': {},
        'negative': {}
    }
    
    for key in ['positive', 'negative']:
        result[key][layer] = [
            act.to('cpu') + factor * steering_vector.to('cpu')
            for act in activations[key][layer]
        ]
    return result


def plot_two_distributions_with_shared_bins(
        pos_proj_orig: np.ndarray, neg_proj_orig: np.ndarray,
        pos_proj_caa: np.ndarray, neg_proj_caa: np.ndarray,
        axes: List[plt.Axes]) -> None:
    """
    Plot two distributions (non-steered and steered) with shared bins and x-axis range.
    This function does not add dataset-specific headers.
    """
    # Determine a common range for the histograms.
    global_min = min(np.min(pos_proj_orig), np.min(neg_proj_orig),
                     np.min(pos_proj_caa), np.min(neg_proj_caa))
    global_max = max(np.max(pos_proj_orig), np.max(neg_proj_orig),
                     np.max(pos_proj_caa), np.max(neg_proj_caa))
    bins = np.linspace(global_min, global_max, 50)
    
    distributions = [
        (pos_proj_orig, neg_proj_orig),
        (pos_proj_caa, neg_proj_caa)
    ]
    labels = ["No Steering", "Steered"]
    
    for (pos_proj, neg_proj), label, ax in zip(distributions, labels, axes):
        ax.hist(neg_proj, bins=bins, alpha=0.5, color='blue', density=True,
                label='Negative', rwidth=0.8)
        ax.hist(pos_proj, bins=bins, alpha=0.5, color='red', density=True,
                label='Positive', rwidth=0.8)
        # Add a small label inside the axis.
        ax.text(0.05, 0.9, label, transform=ax.transAxes, fontsize=8, weight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(global_min, global_max)
        

def main() -> None:
    dataset_idxs = range(36)

    # Constants
    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 500
    TARGET_LAYER = 13
    
    # Setup paths and device
    device = torch.device('cpu')  # Force CPU for all operations
    DATA_PATH = utils.get_path('DATA_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(DATA_PATH, "0_paper_plots")
    os.makedirs(plots_path, exist_ok=True)
    
    num_datasets = len(dataset_idxs)
    # Create a figure with one row per dataset and 2 columns.
    fig, axes = plt.subplots(num_datasets, 2, figsize=(5, 1.5 * num_datasets))
    # If only one dataset is passed, axes may not be a 2D array; ensure it is.
    if num_datasets == 1:
        axes = [axes]  # Wrap the single row in a list.
    
    # Loop over each provided dataset index.
    for row, dataset_idx in enumerate(dataset_idxs):
        # Determine the dataset name from the list (using the first element of the tuple).
        dataset = all_datasets_with_index_and_fraction_anti_steerable_figure_13[dataset_idx][0]
        print(f"\nProcessing dataset: {dataset}")
        
        # Load activations.
        file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
        file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
        activations = load_activations(file_path, NUM_SAMPLES, device)
        
        # 1. Compute original (non-steered) kappa distributions.
        print("Computing original kappa distributions...")
        kappas_dict_pos = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=activations['positive'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        kappas_dict_neg = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=activations['negative'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        pos_proj = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
        neg_proj = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()
        
        # 2. Compute CAA steering vector and apply steering.
        print("Computing CAA steering vector...")
        caa_steering_vector = sv.compute_contrastive_steering_vector_dict(
            positive_activations=activations['positive'],
            negative_activations=activations['negative'],
            device=device
        )[TARGET_LAYER]
        print("Applying CAA steering...")
        caa_steered_activations = {
            'positive': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, -1.0)['positive'],
            'negative': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, 1.0)['negative']
        }
        print("Computing CAA steered kappa distributions...")
        caa_kappas_pos = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=caa_steered_activations['positive'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        caa_kappas_neg = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=caa_steered_activations['negative'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        caa_pos_proj = torch.tensor(caa_kappas_pos[TARGET_LAYER]).numpy()
        caa_neg_proj = torch.tensor(caa_kappas_neg[TARGET_LAYER]).numpy()
        
        # Get the two axes for this dataset (row).
        current_axes = axes[row] if num_datasets > 1 else axes[0]
        # Plot the two distributions (non-steered and steered) on the current row.
        plot_two_distributions_with_shared_bins(
            pos_proj, neg_proj,
            caa_pos_proj, caa_neg_proj,
            current_axes
        )
        
        # Set one common header for this dataset row.
        current_axes[0].set_title(f"Dataset: {dataset}", fontsize=10)
        # Remove any title from the right subplot.
        current_axes[1].set_title("")
    
    # Save the overall figure.
    plt.tight_layout()
    output_file = os.path.join(
        plots_path, 
        f'kappa_distribution_caa_layer_{TARGET_LAYER}_{MODEL_NAME}_multiple_datasets.pdf'
    )
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Successfully saved plot to: {output_file}")
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
