import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
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
        'cohens_d': cohens_d,
        'overlap_coef': overlap_coef,
        'kl_divergence': kl
    }

def load_activations(file_path: str, num_samples: int, device: torch.device) -> dict:
    """Load and prepare activations for kappa calculation."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    return {
        'positive': {
            int(layer): list(data['positive'][layer][:num_samples])
            for layer in data['positive'].keys()
        },
        'negative': {
            int(layer): list(data['negative'][layer][:num_samples])
            for layer in data['negative'].keys()
        }
    }

def main() -> None:
    # Constants
    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 100
    TARGET_LAYER = 13
    
    # Setup paths and device
    device = utils.get_device()
    PROJECT_PATH = utils.get_path('PROJECT_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")
    os.makedirs(plots_path, exist_ok=True)

    # Setup dataset mappings
    datasets = all_datasets_with_index_and_fraction_figure_13_tan_et_al
    dataset_to_index = {name: index for name, index, _ in datasets}

    # Initialize storage for metrics
    metrics_data = {
        'cohens_d': {'x': [], 'y': []},
        'overlap_coef': {'x': [], 'y': []},
        'kl_divergence': {'x': [], 'y': []}
    }

    # Process each dataset
    processed_datasets = []
    for dataset in dataset_to_index.keys():
        try:
            print(f"\nProcessing dataset: {dataset}")
            file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
            file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
            
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
            
            metrics = compute_separation_metrics(pos_proj, neg_proj)
            
            for metric_name in metrics_data.keys():
                metrics_data[metric_name]['x'].append(dataset_to_index[dataset])
                metrics_data[metric_name]['y'].append(metrics[metric_name])
            
            processed_datasets.append(dataset)
            print(f"Successfully processed {dataset}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            continue

    print(f"\nSuccessfully processed datasets: {processed_datasets}")
    print(f"Total processed: {len(processed_datasets)} out of {len(dataset_to_index)}")
    # Create correlation plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metric_titles = {
        'cohens_d': "Cohen's d",
        'overlap_coef': 'Overlap Coefficient',
        'kl_divergence': 'KL Divergence'
    }

    for idx, (metric_name, title) in enumerate(metric_titles.items()):
        x_vals = np.array(metrics_data[metric_name]['x'])
        y_vals = np.array(metrics_data[metric_name]['y'])
        
        # Convert overlap coefficient to percentage
        if metric_name == 'overlap_coef':
            y_vals = y_vals * 100
        
        correlation, p_value = pearsonr(x_vals, y_vals)
        
        axes[idx].scatter(x_vals, y_vals, color='red', marker='s',
                         label=f'Pearson correlation: {correlation:.3f}\np-value: {p_value:.3e}')
        axes[idx].set_xlabel('Dataset Index')
        
        # Adjust y-label for overlap coefficient
        if metric_name == 'overlap_coef':
            axes[idx].set_ylabel(f'{title} (%)')
        else:
            axes[idx].set_ylabel(title)
            
        axes[idx].set_title(f'Dataset Index vs {title}')
        
        # Set legend position based on metric
        if metric_name == 'overlap_coef':
            axes[idx].legend(loc='upper left')
        else:
            axes[idx].legend(loc='lower left')
            
        axes[idx].grid(True)

    plt.tight_layout()
    output_file = os.path.join(plots_path, f'separation_metrics_correlation_{NUM_SAMPLES}.pdf')
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()