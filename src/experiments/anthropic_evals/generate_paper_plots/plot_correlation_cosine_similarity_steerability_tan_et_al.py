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

NUM_SAMPLES = 500
MODEL_NAME = 'llama2_7b_chat'
TARGET_LAYER = 13

load_dotenv()
device = utils.get_device()

# Unpack datasets (each tuple: (dataset_name, index, fraction anti-steerable, avg per sample steerability))
datasets = d.all_datasets_with_fraction_anti_steerable_steerability_figure_13
dataset_to_score = {name: score for name, _, score, _ in datasets}
dataset_to_steerability = {name: steerability for name, _, _, steerability in datasets}

DATA_PATH = utils.get_path('DATA_PATH')
DISK_PATH = utils.get_path('DISK_PATH')
ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
plots_path = os.path.join(DATA_PATH, "0_paper_plots")
intermediate_path = os.path.join(plots_path, "intermediate_data")
os.makedirs(intermediate_path, exist_ok=True)

INTERMEDIATE_JSON = os.path.join(intermediate_path, "layer13_metrics.json")

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

def compute_layer_13_steering_metric(dataset: str, model_name: str = MODEL_NAME) -> float:
    """
    Compute the average cosine similarity between the contrastive activations and the steering vector
    at TARGET_LAYER.
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
    print(f"Available layers in cosine sims: {list(steering_cosine_sims.keys())}")
    
    # Ensure we access the target layer (as int or str)
    layer_key = TARGET_LAYER if TARGET_LAYER in steering_cosine_sims else str(TARGET_LAYER)
    if layer_key not in steering_cosine_sims:
        raise KeyError(f"Layer {TARGET_LAYER} not found in available layers: {list(steering_cosine_sims.keys())}")
    
    # Average the cosine similarities for layer 13
    steering_avg = float(np.mean(steering_cosine_sims[layer_key]))
    return steering_avg

# --- Step 1: Load or compute intermediate data ---
if os.path.exists(INTERMEDIATE_JSON):
    print("Loading intermediate data from JSON file...")
    with open(INTERMEDIATE_JSON, 'r') as f:
        intermediate_data = json.load(f)
else:
    print("Computing intermediate data...")
    intermediate_data = []
    for dataset in dataset_to_score.keys():
        try:
            steering_avg = compute_layer_13_steering_metric(dataset)
            # Store the data for this dataset
            intermediate_data.append({
                "dataset": dataset,
                "avg_per_sample_steerability": dataset_to_steerability[dataset],
                "fraction_anti_steerable": dataset_to_score[dataset] * 100,  # as percentage
                "cosine_similarity": steering_avg
            })
            print(f"Dataset {dataset} processed: cosine_similarity={steering_avg:.3f}")
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
    # Save intermediate data to JSON
    with open(INTERMEDIATE_JSON, 'w') as f:
        json.dump(intermediate_data, f, indent=2)
    print(f"Intermediate data saved to {INTERMEDIATE_JSON}")

# --- Step 2: Prepare data for plotting ---
# Left plot: x = Average per-sample steerability, Right plot: x = Fraction anti-steerable (%)
avg_steerability_vals = np.array([d["avg_per_sample_steerability"] for d in intermediate_data])
fraction_anti_vals = np.array([d["fraction_anti_steerable"] for d in intermediate_data])
cosine_similarity_vals = np.array([d["cosine_similarity"] for d in intermediate_data])

# --- Step 3: Plotting ---
# Create a figure with two subplots: left is avg per-sample steerability, right is fraction anti-steerable (%)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# Left plot: Average per-sample steerability vs cosine similarity
corr_left, pval_left = spearmanr(avg_steerability_vals, cosine_similarity_vals)
axes[0].scatter(avg_steerability_vals, cosine_similarity_vals, color='black', marker='o')
axes[0].set_xlabel("Average per-sample steerability")
axes[0].set_ylabel("Average cosine similarity to SV")
axes[0].set_title("Steerability vs. cosine similarity")
# Create custom legend with two rows of text, no marker, and remove whitespace
custom_handle_left = [Line2D([], [], linestyle='none', label=f"Spearman's ρ: {corr_left:.2f}\np-value: {pval_left:.2e}")]
axes[0].legend(handles=custom_handle_left, loc="lower right", handlelength=0, handletextpad=0)
axes[0].grid(True)

# Right plot: Fraction anti-steerable (%) vs cosine similarity
corr_right, pval_right = spearmanr(fraction_anti_vals, cosine_similarity_vals)
axes[1].scatter(fraction_anti_vals, cosine_similarity_vals, color='black', marker='o')
axes[1].set_xlabel("Fraction of anti-steerable samples (%)")
axes[1].set_ylabel("Average cosine similarity to SV")
axes[1].set_title("Anti-steerable vs. cosine similarity")
# Create custom legend with two rows of text, no marker, and remove whitespace
custom_handle_right = [Line2D([], [], linestyle='none', label=f"Spearman's ρ: {corr_right:.2f}\np-value: {pval_right:.2e}")]
axes[1].legend(handles=custom_handle_right, loc="lower left", handlelength=0, handletextpad=0)
axes[1].grid(True)

plt.tight_layout()

# Save with high resolution (dpi=300) and smaller figure size
plot_file = os.path.join(plots_path, f"iclr2025_figure_layer13_metrics_{NUM_SAMPLES}.pdf")
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"Plot saved to {plot_file}")
