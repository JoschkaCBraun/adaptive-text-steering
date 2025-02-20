import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
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

def plot_distribution_comparison(pos_proj: np.ndarray, neg_proj: np.ndarray,
                               pos_proj_caa: np.ndarray, neg_proj_caa: np.ndarray,
                               axes: List[plt.Axes], titles: List[str]) -> None:
    """Plot original and CAA steered distributions."""
    for idx, (pos, neg, title) in enumerate([(pos_proj, neg_proj, titles[0]), 
                                           (pos_proj_caa, neg_proj_caa, titles[1])]):
        metrics = compute_separation_metrics(pos, neg)
        
        bins = np.linspace(min(np.min(pos), np.min(neg)),
                          max(np.max(pos), np.max(neg)), 40)
        
        axes[idx].hist(neg, bins=bins, alpha=0.5, color='blue', label='Negative')
        axes[idx].hist(pos, bins=bins, alpha=0.5, color='red', label='Positive')
        
        stats_text = (
            f'Positive: μ={metrics["pos_mean"]:.2f}, σ={metrics["pos_std"]:.2f}\n'
            f'Negative: μ={metrics["neg_mean"]:.2f}, σ={metrics["neg_std"]:.2f}\n'
            f"Cohen's d: {metrics['cohens_d']:.2f}\n"
            f'Overlap: {metrics["overlap_coef"]:.2f}\n'
            f'KL div: {metrics["kl_divergence"]:.2f}'
        )
        
        axes[idx].text(0.22, 0.98, stats_text, transform=axes[idx].transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[idx].set_xlabel('Feature strength')
        axes[idx].set_ylabel('Sample count')
        axes[idx].set_title(title)
        axes[idx].legend(fontsize=10)

def load_activations(file_path: str, num_samples: int, device: torch.device) -> dict:
    """Load and prepare activations for kappa calculation."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    # Convert loaded data into the expected format
    result = {
        'positive': {},
        'negative': {}
    }
    
    for class_type in ['positive', 'negative']:
        for layer in data[class_type].keys():
            # Convert to list of tensors
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

def main() -> None:
    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 100
    TARGET_LAYER = 13
    
    device = utils.get_device()
    PROJECT_PATH = utils.get_path('PROJECT_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    datasets = [("corrigible-neutral-HHH", 0, 0),
                ("self-awareness-general-ai", 0, 0),
                ("believes-it-is-not-being-watched-by-humans", 0, 0)]
    
    for dataset, _, _ in datasets:
        file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
        file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
        
        try:
            # Load activations and compute original kappa
            activations = load_activations(file_path, NUM_SAMPLES, device)
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
            
            # Compute CAA steering vector
            caa_steering_vector = sv.compute_contrastive_steering_vector_dict(
                positive_activations=activations['positive'],
                negative_activations=activations['negative'],
                device=device
            )[TARGET_LAYER]
            
            # Apply CAA steering
            caa_steered_activations = {
                'positive': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, -1.0)['positive'],
                'negative': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, 1.0)['negative']
            }
            
            # Compute CAA kappas
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
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            
            # Get projections
            pos_proj = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
            neg_proj = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()
            caa_pos_proj = torch.tensor(caa_kappas_pos[TARGET_LAYER]).numpy()
            caa_neg_proj = torch.tensor(caa_kappas_neg[TARGET_LAYER]).numpy()
            
            # Plot distributions
            plot_distribution_comparison(
                pos_proj, neg_proj,
                caa_pos_proj, caa_neg_proj,
                [ax1, ax2],
                [f'Original Distribution - {dataset}',
                 f'CAA Steered Distribution - {dataset}']
            )
            
            plt.tight_layout()
            output_file = os.path.join(plots_path, f'kappa_distribution_caa_comparison_layer_{TARGET_LAYER}_{MODEL_NAME}_{dataset}.pdf')
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            continue

if __name__ == "__main__":
    main()