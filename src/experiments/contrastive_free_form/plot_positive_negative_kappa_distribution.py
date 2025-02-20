import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.special import kl_div
import torch
import logging
import src.utils.compute_steering_vectors as sv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_separation_metrics(pos_proj: np.ndarray, neg_proj: np.ndarray) -> dict:
    """
    Compute various separation metrics between positive and negative distributions.
    
    Args:
        pos_proj: Projected positive activations
        neg_proj: Projected negative activations
    
    Returns:
        dict: Dictionary containing various separation metrics
    """
    # Calculate means and standard deviations
    pos_mean = np.mean(pos_proj)
    neg_mean = np.mean(neg_proj)
    pos_std = np.std(pos_proj)
    neg_std = np.std(neg_proj)
    
    # Cohen's d
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    cohens_d = (pos_mean - neg_mean) / pooled_std
    
    # Overlapping coefficient (using histogram)
    bins = np.linspace(min(np.min(pos_proj), np.min(neg_proj)),
                      max(np.max(pos_proj), np.max(neg_proj)), 50)
    pos_hist, _ = np.histogram(pos_proj, bins=bins, density=True)
    neg_hist, _ = np.histogram(neg_proj, bins=bins, density=True)
    overlap_coef = np.sum(np.minimum(pos_hist, neg_hist)) * (bins[1] - bins[0])
    
    # KL divergence
    # Add small constant to avoid division by zero
    eps = 1e-10
    pos_hist = pos_hist + eps
    neg_hist = neg_hist + eps
    pos_hist = pos_hist / np.sum(pos_hist)
    neg_hist = neg_hist / np.sum(neg_hist)
    kl = np.sum(kl_div(pos_hist, neg_hist))
    
    return {
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'pos_std': pos_std,
        'neg_std': neg_std,
        'cohens_d': cohens_d,
        'overlap_coef': overlap_coef,
        'kl_divergence': kl
    }


def plot_projection_distributions(pos_proj: np.ndarray, neg_proj: np.ndarray, 
                                ax: plt.Axes, title: str) -> None:
    """
    Plot histograms of positive and negative projections.
    
    Args:
        pos_proj: Projected positive activations
        neg_proj: Projected negative activations
        ax: Matplotlib axes to plot on
        title: Title for the plot
    """
    # Calculate metrics
    metrics = compute_separation_metrics(pos_proj, neg_proj)
    
    # Create histograms
    bins = np.linspace(min(np.min(pos_proj), np.min(neg_proj)),
                      max(np.max(pos_proj), np.max(neg_proj)), 30)
    
    ax.hist(neg_proj, bins=bins, alpha=0.5, color='blue', density=True, label='Negative')
    ax.hist(pos_proj, bins=bins, alpha=0.5, color='red', density=True, label='Positive')
    
    # Add metrics text box
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
    
    ax.set_title(title)
    ax.legend()

def main() -> None:
    # Get paths and device setup from original code
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    DISK_PATH = os.getenv('DISK_PATH')
    if not DISK_PATH:
        raise ValueError("DISK_PATH not found in .env file")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configuration
    model = 'gemma-2-2b-it'
    dataset = 'sentiment_500_en'
    num_samples = 500
    activation_types = ['penultimate', 'last', 'average']
    
    # Load activations
    activations_path = os.path.join(DISK_PATH, "contrastive_free_form_activations")
    base_file_name = f"{model}_activations_{dataset}_for_{num_samples}_samples"
    base_file_path = os.path.join(activations_path, dataset, base_file_name)
    
    # Get all layers from one activation file to determine subplot layout
    with open(f"{base_file_path}_penultimate.pkl", 'rb') as f:
        data = pickle.load(f)
        layers = sorted(list(data['positive'].keys()), key=int)
    
    # Create figure
    fig, axes = plt.subplots(len(layers), len(activation_types), 
                            figsize=(15, 4 * len(layers)))
    # fig.suptitle(f'Projection Distributions for {model} on {dataset}\n{num_samples} samples', 
    #             fontsize=16, y=0.95)
    
    # Process each activation type
    for act_idx, act_type in enumerate(activation_types):
        logging.info(f"Processing {act_type} activations")
        
        # Load activations
        with open(f"{base_file_path}_{act_type}.pkl", 'rb') as f:
            activations = pickle.load(f)
        
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
        
        # Process each layer
        for layer_idx, layer in enumerate(layers):
            logging.info(f"Processing layer {layer}")
            
            pos_proj = np.array(kappas_dict_pos[layer])
            neg_proj = np.array(kappas_dict_neg[layer])
            
            # Plot distributions
            ax = axes[layer_idx, act_idx] if len(layers) > 1 else axes[act_idx]
            plot_projection_distributions(
                pos_proj, neg_proj, ax,
                f"{act_type} - Layer {layer}"
            )
    
    # Adjust layout and save
    plt.tight_layout()
    plots_path = os.path.join(DATA_PATH, "contrastive_free_form_data", "plots")
    os.makedirs(plots_path, exist_ok=True)
    output_file = os.path.join(
        plots_path, 
        f'positive_negative_kappa_distribution_{model}_{dataset}_{num_samples}.pdf'
    )
    plt.savefig(output_file, bbox_inches='tight')
    logging.info(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()