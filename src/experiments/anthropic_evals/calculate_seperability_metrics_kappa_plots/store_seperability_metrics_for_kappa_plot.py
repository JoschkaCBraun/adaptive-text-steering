"""
plot_kappa_distribution_for_steered_and_non_steered_activations.py

This file has been adapted to compute separation metrics (overlap coefficient,
d-prime, and ROC AUC) for the non-steered kappa projections for each dataset.
The resulting statistics for each dataset are stored in a JSON file in
'0_paper_plots/intermediate_data/'.
"""

import os
import pickle
import json
import numpy as np
from typing import List
import torch

from sklearn.metrics import roc_auc_score

from src.utils.dataset_names import all_datasets_with_index_and_fraction_anti_steerable_figure_13
import src.utils.compute_steering_vectors as sv
import src.utils as utils


def compute_separation_metrics(pos_proj: np.ndarray, neg_proj: np.ndarray) -> dict:
    """
    Compute statistical separation metrics between positive and negative projections.
    The metrics computed are:
      - d_prime: normalized difference between means,
      - overlap_coef: overlapping coefficient of the two distributions,
      - roc_auc: area under the ROC curve when using kappa scores as decision values.
    """
    # Compute d-prime (using the same formula as Cohen's d)
    pos_mean, neg_mean = np.mean(pos_proj), np.mean(neg_proj)
    pos_std, neg_std = np.std(pos_proj), np.std(neg_proj)
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    d_prime = (pos_mean - neg_mean) / pooled_std

    # Compute overlapping coefficient using histograms
    bins = np.linspace(min(np.min(pos_proj), np.min(neg_proj)),
                       max(np.max(pos_proj), np.max(neg_proj)), 50)
    pos_hist, _ = np.histogram(pos_proj, bins=bins, density=True)
    neg_hist, _ = np.histogram(neg_proj, bins=bins, density=True)
    overlap_coef = np.sum(np.minimum(pos_hist, neg_hist)) * (bins[1] - bins[0])

    # Compute ROC AUC:
    labels = np.concatenate([np.ones_like(pos_proj), np.zeros_like(neg_proj)])
    scores = np.concatenate([pos_proj, neg_proj])
    try:
        roc_auc = roc_auc_score(labels, scores)
    except Exception as e:
        roc_auc = None

    return {
         'd_prime': d_prime,
         'overlap_coef': overlap_coef,
         'roc_auc': roc_auc
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
    intermediate_data_path = os.path.join(plots_path, "intermediate_data")
    os.makedirs(intermediate_data_path, exist_ok=True)
    
    results = []  # This list will store the metrics for each dataset

    # Loop over each provided dataset index.
    for dataset_idx in dataset_idxs:
        # Determine the dataset name from the list.
        dataset = all_datasets_with_index_and_fraction_anti_steerable_figure_13[dataset_idx][0]
        print(f"\nProcessing dataset: {dataset}")
        
        # Load activations.
        file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
        file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
        activations = load_activations(file_path, NUM_SAMPLES, device)
        
        # Compute original (non-steered) kappa distributions.
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
        
        # Compute the separation metrics on the non-steered distributions.
        metrics = compute_separation_metrics(pos_proj, neg_proj)
        print(f"Metrics for dataset {dataset}: {metrics}")

        # Store the results in the desired dictionary format.
        dataset_results = {
            "dataset": dataset,
            "overlap_coef": metrics['overlap_coef'],
            "d_prime": metrics['d_prime'],
            "roc_auc": metrics['roc_auc']
        }
        results.append(dataset_results)
    
    # Save the results in a JSON file.
    output_file = os.path.join(
        intermediate_data_path,
        f'separation_metrics_non_steered_{MODEL_NAME}.json'
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved separation metrics to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
