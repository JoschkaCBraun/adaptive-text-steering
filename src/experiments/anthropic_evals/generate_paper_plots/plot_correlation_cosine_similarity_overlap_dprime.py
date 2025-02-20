import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D

import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa
import src.utils as utils
import src.utils.dataset_names as d

# Provided function to compute pairwise cosine similarity (not used here)
def compute_pairwise_cosine_similarity(activation_list, device: torch.device = None) -> list:
    """
    Compute pairwise cosine similarity between all activation vectors.
    Returns a list of unique similarities (no duplicates).
    """
    if device is None:
        device = activation_list[0].device

    vectors_stacked = torch.stack([activation.to(device) for activation in activation_list])
    vectors_normalized = F.normalize(vectors_stacked, p=2, dim=1)
    similarity_matrix = torch.mm(vectors_normalized, vectors_normalized.T)

    mask = torch.ones_like(similarity_matrix, dtype=torch.bool).triu(diagonal=1)
    similarities = similarity_matrix[mask].tolist()
    return similarities

NUM_SAMPLES = 500
MODEL_NAME = 'llama2_7b_chat'
TARGET_LAYER = 13

load_dotenv()
device = utils.get_device()

# Unpack datasets (each tuple: (dataset_name, index, fraction anti-steerable, avg per sample steerability))
# (Note: we will not use the steerability and anti-steerable fractions for plotting anymore)
datasets = d.all_datasets_with_fraction_anti_steerable_steerability_figure_13
dataset_names = [name for name, _, _, _ in datasets]

DATA_PATH = utils.get_path('DATA_PATH')
DISK_PATH = utils.get_path('DISK_PATH')
ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
plots_path = os.path.join(DATA_PATH, "0_paper_plots")
intermediate_path = os.path.join(plots_path, "intermediate_data")
os.makedirs(intermediate_path, exist_ok=True)

INTERMEDIATE_JSON = os.path.join(intermediate_path, "layer13_metrics_overlap_dprime.json")

def load_activations(file_path: str) -> dict:
    print(f"Loading activations from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return {
        'positive': {int(layer): list(data['positive'][layer][:NUM_SAMPLES])
                     for layer in data['positive'].keys()},
        'negative': {int(layer): list(data['negative'][layer][:NUM_SAMPLES])
                     for layer in data['negative'].keys()}
    }

def compute_layer_13_metrics(dataset: str, model_name: str = MODEL_NAME) -> dict:
    """
    Compute the average cosine similarity between the contrastive activations and the steering vector at TARGET_LAYER,
    and compute two new metrics (overlap coefficient and d-prime) based on the projection of activations onto the
    difference-of-means (steering) vector.
    """
    print(f"\nProcessing dataset: {dataset}")
    file_name = f"{model_name}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
    file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")
    
    # Load activations
    activations = load_activations(file_path)
    
    # Compute contrastive activations (usually the difference between positive and negative)
    print("Computing contrastive activations...")
    contrastive_activations = sv.compute_contrastive_activations(activations['positive'],
                                                                 activations['negative'],
                                                                 device)
    
    # Compute steering vectors for each layer
    print("Computing steering vectors...")
    steering_vectors = sv.compute_contrastive_steering_vector_dict(activations['positive'],
                                                                   activations['negative'],
                                                                   device)
    
    # Compute cosine similarity between each contrastive activation and its steering vector
    print("Computing cosine similarities between contrastive activations and steering vector...")
    steering_cosine_sims = cpa.compute_many_against_one_cosine_similarity_dict(contrastive_activations,
                                                                              steering_vectors,
                                                                              device)
    # Access the target layer (as int or str)
    layer_key = TARGET_LAYER if TARGET_LAYER in steering_cosine_sims else str(TARGET_LAYER)
    if layer_key not in steering_cosine_sims:
        raise KeyError(f"Layer {TARGET_LAYER} not found in available layers: {list(steering_cosine_sims.keys())}")
    cosine_avg = float(np.mean(steering_cosine_sims[layer_key]))
    print(f"Cosine similarity (avg) for dataset {dataset}: {cosine_avg:.3f}")
    
    # --- Compute projections onto the difference-of-means (steering) line ---
    # We use the provided kappa function to project activations.
    print("Computing projections (kappa) for positive activations...")
    kappas_dict_pos = sv.calculate_feature_expressions_kappa_dict(
        activations_dict=activations['positive'],
        positive_activations_dict=activations['positive'],
        negative_activations_dict=activations['negative'],
        device=device
    )
    print("Computing projections (kappa) for negative activations...")
    kappas_dict_neg = sv.calculate_feature_expressions_kappa_dict(
        activations_dict=activations['negative'],
        positive_activations_dict=activations['positive'],
        negative_activations_dict=activations['negative'],
        device=device
    )
    # Extract projections for TARGET_LAYER
    pos_proj = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
    neg_proj = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()
    
    # --- Compute d-prime ---
    mean_pos = np.mean(pos_proj)
    mean_neg = np.mean(neg_proj)
    var_pos = np.var(pos_proj)
    var_neg = np.var(neg_proj)
    d_prime = (mean_pos - mean_neg) / np.sqrt((var_pos + var_neg) / 2)
    print(f"d-prime for dataset {dataset}: {d_prime:.3f}")
    
    # --- Compute overlap coefficient ---
    def compute_overlap_coefficient(proj1: np.ndarray, proj2: np.ndarray, bins: int = 50) -> float:
        # Use a common range for both distributions
        global_min = min(np.min(proj1), np.min(proj2))
        global_max = max(np.max(proj1), np.max(proj2))
        hist1, edges = np.histogram(proj1, bins=bins, range=(global_min, global_max), density=True)
        hist2, _ = np.histogram(proj2, bins=bins, range=(global_min, global_max), density=True)
        bin_width = edges[1] - edges[0]
        overlap = np.sum(np.minimum(hist1, hist2)) * bin_width
        return overlap
    
    overlap_coef = compute_overlap_coefficient(pos_proj, neg_proj)
    print(f"Overlap coefficient for dataset {dataset}: {overlap_coef:.3f}")
    
    # Return all computed metrics
    return {
        "dataset": dataset,
        "cosine_similarity": cosine_avg,
        "overlap_coefficient": overlap_coef,
        "d_prime": d_prime
    }

# --- Step 1: Load or compute intermediate data ---
if os.path.exists(INTERMEDIATE_JSON):
    print("Loading intermediate data from JSON file...")
    with open(INTERMEDIATE_JSON, 'r') as f:
        intermediate_data = json.load(f)
else:
    print("Computing intermediate data for each dataset...")
    intermediate_data = []
    for dataset in dataset_names:
        try:
            metrics = compute_layer_13_metrics(dataset)
            intermediate_data.append(metrics)
            print(f"Dataset {dataset} processed.")
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
    # Save intermediate data to JSON
    with open(INTERMEDIATE_JSON, 'w') as f:
        json.dump(intermediate_data, f, indent=2)
    print(f"Intermediate data saved to {INTERMEDIATE_JSON}")

# --- Step 2: Prepare data for plotting ---
cosine_similarity_vals = np.array([d["cosine_similarity"] for d in intermediate_data])
overlap_coef_vals = np.array([d["overlap_coefficient"] for d in intermediate_data])
d_prime_vals = np.array([d["d_prime"] for d in intermediate_data])

# --- Step 3: Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# Left plot: Overlap Coefficient vs cosine similarity
corr_left, pval_left = spearmanr(overlap_coef_vals, cosine_similarity_vals)
axes[0].scatter(overlap_coef_vals, cosine_similarity_vals, color='black', marker='o')
axes[0].set_xlabel("Overlap Coefficient")
axes[0].set_ylabel("Average Cosine Similarity to SV")
axes[0].set_title("Cosine Similarity vs. Overlap")
custom_handle_left = [Line2D([], [], linestyle='none', label=f"Spearman's ρ: {corr_left:.2f}\np-value: {pval_left:.2e}")]
axes[0].legend(handles=custom_handle_left, loc="lower right", handlelength=0, handletextpad=0)
axes[0].grid(True)

# Right plot: d-prime vs cosine similarity
corr_right, pval_right = spearmanr(d_prime_vals, cosine_similarity_vals)
axes[1].scatter(d_prime_vals, cosine_similarity_vals, color='black', marker='o')
axes[1].set_xlabel("d-prime")
axes[1].set_ylabel("Average Cosine Similarity to SV")
axes[1].set_title("Cosine Similarity vs. d-prime")
custom_handle_right = [Line2D([], [], linestyle='none', label=f"Spearman's ρ: {corr_right:.2f}\np-value: {pval_right:.2e}")]
axes[1].legend(handles=custom_handle_right, loc="lower left", handlelength=0, handletextpad=0)
axes[1].grid(True)

plt.tight_layout()

# Save with high resolution and a new file name reflecting the new content
plot_file = os.path.join(plots_path, f"iclr2025_figure_layer13_metrics_overlap_dprime_{NUM_SAMPLES}.pdf")
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"Plot saved to {plot_file}")
