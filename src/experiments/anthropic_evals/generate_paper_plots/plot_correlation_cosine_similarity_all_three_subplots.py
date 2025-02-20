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

titlesize = 10
legendsize = 9

# --- Provided utility function for pairwise cosine similarity (not used here) ---
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

# --- Global parameters ---
NUM_SAMPLES = 500
MODEL_NAME = 'llama2_7b_chat'
TARGET_LAYER = 13

load_dotenv()
device = utils.get_device()

# --- Get dataset metadata ---
# Each tuple in datasets is: (dataset_name, index, fraction_anti_steerable, avg_per_sample_steerability)
datasets = d.all_datasets_with_fraction_anti_steerable_steerability_figure_13
# Create dictionaries to quickly look up values for each dataset
dataset_to_score = {name: score for name, _, score, _ in datasets}
dataset_to_steerability = {name: steerability for name, _, _, steerability in datasets}
dataset_names = [name for name, _, _, _ in datasets]

# --- Define paths ---
DATA_PATH = utils.get_path('DATA_PATH')
DISK_PATH = utils.get_path('DISK_PATH')
ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
plots_path = os.path.join(DATA_PATH, "0_paper_plots")
intermediate_path = os.path.join(plots_path, "intermediate_data")
os.makedirs(intermediate_path, exist_ok=True)
INTERMEDIATE_JSON = os.path.join(intermediate_path, "three_metrics_all_in_one.json")

# --- Helper function to load activations ---
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

# --- Compute metrics (cosine similarity and d-prime) for a given dataset ---
def compute_layer_13_metrics(dataset: str, model_name: str = MODEL_NAME) -> dict:
    """
    For the given dataset, compute:
      - Cosine similarity between the contrastive activations and the steering vector (at TARGET_LAYER).
      - d-prime based on the projection (using the provided kappa calculation).
    
    The method also returns the overlap coefficient (optional) but here we only need d-prime.
    """
    print(f"\nProcessing dataset: {dataset}")
    file_name = f"{model_name}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
    file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")
    
    # Load activations
    activations = load_activations(file_path)
    
    # --- Compute contrastive activations and steering vectors ---
    print("Computing contrastive activations...")
    contrastive_activations = sv.compute_contrastive_activations(
        activations['positive'],
        activations['negative'],
        device
    )
    print("Computing steering vectors...")
    steering_vectors = sv.compute_contrastive_steering_vector_dict(
        activations['positive'],
        activations['negative'],
        device
    )
    
    # --- Compute cosine similarities between contrastive activations and the steering vector ---
    print("Computing cosine similarities between contrastive activations and steering vector...")
    steering_cosine_sims = cpa.compute_many_against_one_cosine_similarity_dict(
        contrastive_activations,
        steering_vectors,
        device
    )
    
    # Ensure we access the target layer (key may be int or str)
    layer_key = TARGET_LAYER if TARGET_LAYER in steering_cosine_sims else str(TARGET_LAYER)
    if layer_key not in steering_cosine_sims:
        raise KeyError(f"Layer {TARGET_LAYER} not found in available layers: {list(steering_cosine_sims.keys())}")
    cosine_avg = float(np.mean(steering_cosine_sims[layer_key]))
    print(f"Cosine similarity (avg) for dataset {dataset}: {cosine_avg:.3f}")
    
    # --- Compute d-prime ---
    # Use the provided kappa method to project activations onto the difference-of-means (steering) line.
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
    # Extract projections for the TARGET_LAYER
    pos_proj = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
    neg_proj = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()
    
    mean_pos = np.mean(pos_proj)
    mean_neg = np.mean(neg_proj)
    var_pos = np.var(pos_proj)
    var_neg = np.var(neg_proj)
    d_prime = (mean_pos - mean_neg) / np.sqrt((var_pos + var_neg) / 2)
    print(f"d-prime for dataset {dataset}: {d_prime:.3f}")
    
    # Return computed metrics for this dataset
    return {
        "dataset": dataset,
        "cosine_similarity": cosine_avg,
        "d_prime": d_prime
    }

# --- Step 1: Load or compute intermediate data for all datasets ---
if os.path.exists(INTERMEDIATE_JSON):
    print("Loading intermediate data from JSON file...")
    with open(INTERMEDIATE_JSON, 'r') as f:
        intermediate_data = json.load(f)
else:
    print("Computing intermediate data for each dataset...")
    intermediate_data = []
    for dataset in dataset_names:
        try:
            # Compute cosine similarity and d-prime using the function above.
            metrics = compute_layer_13_metrics(dataset)
            # Add extra metadata from our dataset dictionaries
            metrics["avg_per_sample_steerability"] = dataset_to_steerability[dataset]
            # Convert fraction to percentage
            metrics["fraction_anti_steerable"] = dataset_to_score[dataset] * 100  
            intermediate_data.append(metrics)
            print(f"Dataset {dataset} processed.")
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
    # Save intermediate data to JSON for future use.
    with open(INTERMEDIATE_JSON, 'w') as f:
        json.dump(intermediate_data, f, indent=2)
    print(f"Intermediate data saved to {INTERMEDIATE_JSON}")

# --- Step 2: Prepare arrays for plotting ---
# Extract values for each metric
avg_steerability_vals = np.array([d["avg_per_sample_steerability"] for d in intermediate_data])
fraction_anti_vals = np.array([d["fraction_anti_steerable"] for d in intermediate_data])
cosine_similarity_vals = np.array([d["cosine_similarity"] for d in intermediate_data])
d_prime_vals = np.array([d["d_prime"] for d in intermediate_data])

# --- Step 3: Create a figure with three subplots ---
fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # Adjust figure size as needed

# Subplot 1: Cosine similarity vs. Mean per-sample steerability
corr1, pval1 = spearmanr(avg_steerability_vals, cosine_similarity_vals)
axes[0].scatter(avg_steerability_vals, cosine_similarity_vals, color='black', marker='o')
axes[0].set_xlabel("Mean per-sample steerability")
axes[0].set_ylabel("Mean cosine similarity\nto steering vector")
axes[0].set_title("Cosine similarity vs. steerability", fontsize=titlesize)
custom_handle1 = [Line2D([], [], linestyle='none', label=f"Spearman's ρ: {corr1:.2f}\np-value: {pval1:.2e}")]
axes[0].legend(handles=custom_handle1, loc="lower right", handlelength=0, handletextpad=0, fontsize=legendsize)
axes[0].grid(True)

# Subplot 2: Cosine similarity vs. fraction anti-steerable (%)
corr2, pval2 = spearmanr(fraction_anti_vals, cosine_similarity_vals)
axes[1].scatter(fraction_anti_vals, cosine_similarity_vals, color='black', marker='o')
axes[1].set_xlabel("Fraction anti-steerable (%)")
# axes[1].set_ylabel("Mean cosine similarity to steering vector")
axes[1].set_title("Cosine similarity vs. anti-steerable", fontsize=titlesize)
custom_handle2 = [Line2D([], [], linestyle='none', label=f"Spearman's ρ: {corr2:.2f}\np-value: {pval2:.2e}")]
axes[1].legend(handles=custom_handle2, loc="lower left", handlelength=0, handletextpad=0, fontsize=legendsize)
axes[1].grid(True)

# Subplot 3: Cosine similarity vs. d-prime
corr3, pval3 = spearmanr(d_prime_vals, cosine_similarity_vals)
axes[2].scatter(d_prime_vals, cosine_similarity_vals, color='black', marker='o')
axes[2].set_xlabel("Discriminability index (d')")
# axes[2].set_ylabel("Mean cosine similarity to steering vector")
axes[2].set_title("Cosine similarity vs. discriminability", fontsize=titlesize)

custom_handle3 = [Line2D([], [], linestyle='none', label=f"Spearman's ρ: {corr3:.2f}\np-value: {pval3:.2e}")]
axes[2].legend(handles=custom_handle3, loc="lower right", handlelength=0, handletextpad=0, fontsize=legendsize)
axes[2].grid(True)

plt.tight_layout()

# --- Step 4: Save the plot to a PDF file ---
plot_file = os.path.join(plots_path, f"three_subplots_cosine_similarity_{NUM_SAMPLES}.pdf")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to {plot_file}")
