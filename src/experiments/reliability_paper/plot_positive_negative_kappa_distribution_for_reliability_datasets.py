import os
import pickle
import numpy as np
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

def plot_distribution(pos_proj: np.ndarray, neg_proj: np.ndarray, 
                     ax: plt.Axes, title: str) -> None:
    """Plot distribution of positive and negative projections with metrics."""
    metrics = compute_separation_metrics(pos_proj, neg_proj)
    
    bins = np.linspace(min(np.min(pos_proj), np.min(neg_proj)),
                      max(np.max(pos_proj), np.max(neg_proj)), 40)
    
    ax.hist(neg_proj, bins=bins, alpha=0.5, color='blue', label='Negative')
    ax.hist(pos_proj, bins=bins, alpha=0.5, color='red', label='Positive')
    
    stats_text = (
        f'Positive: μ={metrics["pos_mean"]:.2f}, σ={metrics["pos_std"]:.2f}\n'
        f'Negative: μ={metrics["neg_mean"]:.2f}, σ={metrics["neg_std"]:.2f}\n'
        f"Cohen's d: {metrics['cohens_d']:.2f}\n"
        f'Overlap: {metrics["overlap_coef"]:.2f}\n'
        f'KL div: {metrics["kl_divergence"]:.2f}'
    )
    
    # Center the stats text and increase font size
    ax.text(0.5, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='center', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set axis labels and title
    ax.set_xlabel('Feature strength (projected value on difference of means line)')
    ax.set_ylabel('Sample count')
    ax.set_title(f'Feature strength distribution for {title}, Llama2 7b chat (n=500)',
                 pad=10)
    
    # Increase legend size
    ax.legend(fontsize=10)

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

def main() -> None:
    # Constants
    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 100
    TARGET_LAYER = 13
    
    # Setup paths
    device = utils.get_device()
    PROJECT_PATH = utils.get_path('PROJECT_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")
    os.makedirs(plots_path, exist_ok=True)

    # Get sorted datasets
    datasets = all_datasets_with_index_and_fraction_figure_13_tan_et_al

    
    # Create figure with subplots
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 4 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    
    # Process each dataset
    for idx, (dataset, _, _) in enumerate(datasets):
        print(f"\nProcessing dataset: {dataset}")
        file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
        file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
        
        try:
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
            
            pos_proj = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
            neg_proj = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()
            
            plot_distribution(pos_proj, neg_proj, axes[idx], f"dataset: {dataset}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            continue
    
    # Save the plot
    plt.tight_layout()
    output_file = os.path.join(plots_path, f'kappa_distribution_layer_{TARGET_LAYER}_{MODEL_NAME}.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()