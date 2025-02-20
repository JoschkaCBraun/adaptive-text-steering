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
from src.utils.dataset_names import all_datasets_with_index_and_fraction_figure_13_tan_et_al
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

def plot_four_distributions_with_shared_bins(
        pos_proj_orig: np.ndarray, neg_proj_orig: np.ndarray,
        pos_proj_caa: np.ndarray, neg_proj_caa: np.ndarray,
        pos_proj_icv: np.ndarray, neg_proj_icv: np.ndarray,
        pos_proj_mimic: np.ndarray, neg_proj_mimic: np.ndarray,
        axes: List[plt.Axes], dataset: str) -> None:
    """Plot four distributions with shared bins and x-axis range."""
    
    # Find global min and max across all distributions
    print("Computing global min/max...")
    global_min = min(
        np.min(pos_proj_orig), np.min(neg_proj_orig),
        np.min(pos_proj_caa), np.min(neg_proj_caa),
        np.min(pos_proj_icv), np.min(neg_proj_icv),
        np.min(pos_proj_mimic), np.min(neg_proj_mimic)
    )
    global_max = max(
        np.max(pos_proj_orig), np.max(neg_proj_orig),
        np.max(pos_proj_caa), np.max(neg_proj_caa),
        np.max(pos_proj_icv), np.max(neg_proj_icv),
        np.max(pos_proj_mimic), np.max(neg_proj_mimic)
    )
    print(f"Global range: [{global_min:.3f}, {global_max:.3f}]")
    
    # Create shared bins
    bins = np.linspace(global_min, global_max, 50)
    
    # Data to plot
    distributions = [
        (pos_proj_orig, neg_proj_orig, "Original Kappa Distribution"),
        (pos_proj_caa, neg_proj_caa, "CAA Steered Distribution"),
        (pos_proj_icv, neg_proj_icv, "ICV Steered Distribution"),
        (pos_proj_mimic, neg_proj_mimic, "MiMiC Steered Distribution")
    ]
    
    # Plot each distribution
    for (pos_proj, neg_proj, title), ax in zip(distributions, axes):
        print(f"Plotting {title}...")
        metrics = compute_separation_metrics(pos_proj, neg_proj)
        
        ax.hist(neg_proj, bins=bins, alpha=0.5, color='blue', density=True, label='Negative')
        ax.hist(pos_proj, bins=bins, alpha=0.5, color='red', density=True, label='Positive')
        
        stats_text = (
            f'Positive: μ={metrics["pos_mean"]:.3f}, σ={metrics["pos_std"]:.3f}\n'
            f'Negative: μ={metrics["neg_mean"]:.3f}, σ={metrics["neg_std"]:.3f}\n'
            f"Cohen's d: {metrics['cohens_d']:.3f}\n"
            f'Overlap: {metrics["overlap_coef"]:.3f}\n'
            f'KL div: {metrics["kl_divergence"]:.3f}'
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f"{title} - Dataset: {dataset}")
        ax.legend()
        ax.set_xlim(global_min, global_max)

def main() -> None:
    print("Starting main function...")
    # Constants
    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 500
    TARGET_LAYER = 13
    
    # Setup paths and device
    device = torch.device('cpu')  # Force CPU for all operations
    PROJECT_PATH = utils.get_path('PROJECT_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")
    os.makedirs(plots_path, exist_ok=True)

    # Get first dataset
    dataset = all_datasets_with_index_and_fraction_figure_13_tan_et_al[0][0]
    print(f"\nProcessing dataset: {dataset}")
    
    # Load activations
    file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
    file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
    activations = load_activations(file_path, NUM_SAMPLES, device)
    
    # Create figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

    # 1. Calculate original kappa distributions
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
    
    # Get original projections
    pos_proj = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
    neg_proj = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()
    
    # 2. Compute steering vectors
    print("Computing steering vectors...")
    caa_steering_vector = sv.compute_contrastive_steering_vector_dict(
        positive_activations=activations['positive'],
        negative_activations=activations['negative'],
        device=device
    )[TARGET_LAYER]
    
    icv_steering_vectors = sv.compute_paired_icv_steering_vector(
        positive_activations=activations['positive'],
        negative_activations=activations['negative'],
        device=device
    )
    
    mimic_steering_vectors = sv.compute_mimic_steering_vector(
        positive_activations=activations['positive'],
        negative_activations=activations['negative'],
        device=device
    )
    
    # 3. Apply steering methods
    print("Applying steering methods...")
    # CAA
    caa_steered_activations = {
        'positive': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, -1.0)['positive'],
        'negative': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, 1.0)['negative']
    }
    
    # ICV
    icv_steered_pos = sv.apply_paired_icv_steering(
        {TARGET_LAYER: activations['positive'][TARGET_LAYER]}, 
        {TARGET_LAYER: icv_steering_vectors[TARGET_LAYER]}, 
        device
    )
    icv_steered_neg = sv.apply_paired_icv_steering(
        {TARGET_LAYER: activations['negative'][TARGET_LAYER]}, 
        {TARGET_LAYER: icv_steering_vectors[TARGET_LAYER]}, 
        device
    )
    icv_steered_activations = {
        'positive': {TARGET_LAYER: icv_steered_pos[TARGET_LAYER]},
        'negative': {TARGET_LAYER: icv_steered_neg[TARGET_LAYER]}
    }
    
    # MiMiC
    mimic_steered_pos = sv.apply_mimic_steering(
        {TARGET_LAYER: activations['positive'][TARGET_LAYER]}, 
        {TARGET_LAYER: mimic_steering_vectors[TARGET_LAYER]}, 
        device
    )
    mimic_steered_neg = sv.apply_mimic_steering(
        {TARGET_LAYER: activations['negative'][TARGET_LAYER]}, 
        {TARGET_LAYER: mimic_steering_vectors[TARGET_LAYER]}, 
        device
    )
    mimic_steered_activations = {
        'positive': {TARGET_LAYER: mimic_steered_pos[TARGET_LAYER]},
        'negative': {TARGET_LAYER: mimic_steered_neg[TARGET_LAYER]}
    }
    
    # 4. Calculate kappas for steered activations
    print("Computing steered kappa distributions...")
    # CAA
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

    # ICV
    icv_kappas_pos = sv.calculate_feature_expressions_kappa_dict(
        activations_dict=icv_steered_activations['positive'],
        positive_activations_dict=activations['positive'],
        negative_activations_dict=activations['negative'],
        device=device
    )
    icv_kappas_neg = sv.calculate_feature_expressions_kappa_dict(
        activations_dict=icv_steered_activations['negative'],
        positive_activations_dict=activations['positive'],
        negative_activations_dict=activations['negative'],
        device=device
    )

    # MiMiC
    mimic_kappas_pos = sv.calculate_feature_expressions_kappa_dict(
        activations_dict=mimic_steered_activations['positive'],
        positive_activations_dict=activations['positive'],
        negative_activations_dict=activations['negative'],
        device=device
    )
    mimic_kappas_neg = sv.calculate_feature_expressions_kappa_dict(
        activations_dict=mimic_steered_activations['negative'],
        positive_activations_dict=activations['positive'],
        negative_activations_dict=activations['negative'],
        device=device
    )

    # 5. Get projections for plotting
    caa_pos_proj = torch.tensor(caa_kappas_pos[TARGET_LAYER]).numpy()
    caa_neg_proj = torch.tensor(caa_kappas_neg[TARGET_LAYER]).numpy()
    icv_pos_proj = torch.tensor(icv_kappas_pos[TARGET_LAYER]).numpy()
    icv_neg_proj = torch.tensor(icv_kappas_neg[TARGET_LAYER]).numpy()
    mimic_pos_proj = torch.tensor(mimic_kappas_pos[TARGET_LAYER]).numpy()
    mimic_neg_proj = torch.tensor(mimic_kappas_neg[TARGET_LAYER]).numpy()

    # 6. Plot distributions
    print("\nCreating plots...")
    plot_four_distributions_with_shared_bins(
        pos_proj, neg_proj,
        caa_pos_proj, caa_neg_proj,
        icv_pos_proj, icv_neg_proj,
        mimic_pos_proj, mimic_neg_proj,
        [ax1, ax2, ax3, ax4],
        dataset
    )
    
    # Save plot
    print("\nSaving plot...")
    plt.tight_layout()
    output_file = os.path.join(
        plots_path, 
        f'kappa_distribution_all_methods_layer_{TARGET_LAYER}_{MODEL_NAME}_{dataset}.pdf'
    )
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Successfully saved plot to: {output_file}")
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()