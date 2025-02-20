import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr  # using Spearman's correlation
from dotenv import load_dotenv
import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa
import src.utils as utils

NUM_SAMPLES = 500
MODEL_NAME = 'llama2_7b_chat'
TARGET_LAYER = 13

load_dotenv()
device = utils.get_device()

DATA_PATH = utils.get_path('DATA_PATH')
DISK_PATH = utils.get_path('DISK_PATH')
ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
plots_path = os.path.join(DATA_PATH, "0_paper_plots")
# Create intermediate data directory within the paper plots folder.
intermediate_dir = os.path.join(plots_path, "intermediate_data")
os.makedirs(intermediate_dir, exist_ok=True)
# Descriptive JSON file name.
intermediate_json_filename = f"intermediate_data_steerability_metrics_layer13_similarity_{NUM_SAMPLES}.json"
intermediate_json_filepath = os.path.join(intermediate_dir, intermediate_json_filename)

# New: Load the JSON file containing steerability metrics.
metrics_file_path = os.path.join(DATA_PATH, "anthropic_evals_results", "steerability_metrics", "steerability_metrics.json")
if not os.path.exists(metrics_file_path):
    raise FileNotFoundError(f"Metrics JSON file not found: {metrics_file_path}")

with open(metrics_file_path, 'r') as f:
    metrics_data = json.load(f)

def load_activations(file_path: str) -> dict:
    print(f"Loading activations from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return {
        'positive': {
            int(layer): list(data['positive'][layer][:NUM_SAMPLES])
            for layer in data['positive'].keys()
        },
        'negative': {
            int(layer): list(data['negative'][layer][:NUM_SAMPLES])
            for layer in data['negative'].keys()
        }
    }

def compute_layer_13_similarity(dataset: str, model_name: str = MODEL_NAME) -> float:
    print(f"\nProcessing dataset: {dataset}")
    file_name = f"{model_name}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
    file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")
    
    activations = load_activations(file_path)
    print("Computing contrastive activations...")
    contrastive_activations = sv.compute_contrastive_activations(activations['positive'], activations['negative'], device)
    print("Computing steering vectors...")
    steering_vectors = sv.compute_contrastive_steering_vector_dict(activations['positive'], activations['negative'], device)
    
    print("Computing cosine similarities...")
    steering_cosine_sims = cpa.compute_many_against_one_cosine_similarity_dict(contrastive_activations, steering_vectors, device)
    print(f"Available layers in cosine sims: {list(steering_cosine_sims.keys())}")
    
    layer_key = TARGET_LAYER if TARGET_LAYER in steering_cosine_sims else str(TARGET_LAYER)
    if layer_key not in steering_cosine_sims:
        raise KeyError(f"Layer {TARGET_LAYER} not found in available layers: {list(steering_cosine_sims.keys())}")
    
    return float(np.mean(steering_cosine_sims[layer_key]))

# Check if the intermediate data JSON already exists.
if os.path.exists(intermediate_json_filepath):
    print(f"Loading intermediate data from {intermediate_json_filepath}")
    with open(intermediate_json_filepath, 'r') as f:
        intermediate_data = json.load(f)
else:
    print("Intermediate data file not found; computing intermediate data now.")
    # Prepare arrays to store data for plotting.
    intermediate_data = {"datasets": []}

    # Iterate over each dataset in the metrics JSON.
    for dataset, prompt_data in metrics_data.items():
        # Ensure the expected "prefilled_answer" key exists.
        if "prefilled_answer" not in prompt_data:
            raise KeyError(f"Dataset '{dataset}' is missing the 'prefilled_answer' key in the metrics JSON.")
        
        # Extract metrics from the "prefilled_answer" prompt type.
        average_metrics = prompt_data["prefilled_answer"]
        if "anti_steerable_percentage" not in average_metrics or "mean_logit_diff" not in average_metrics:
            raise KeyError(f"Dataset '{dataset}' does not contain the required keys in the 'prefilled_answer' section.")
        
        try:
            similarity = compute_layer_13_similarity(dataset)
            entry = {
                "dataset": dataset,
                "anti_steerable_percentage": average_metrics["anti_steerable_percentage"],
                "mean_logit_diff": average_metrics["mean_logit_diff"],
                "layer13_similarity": similarity
            }
            intermediate_data["datasets"].append(entry)
            print(f"Successfully processed {dataset}, similarity: {similarity:.3f}")
        except Exception as e:
            print(f"Error processing dataset {dataset}")
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e)}")
    
    print(f"\nTotal processed datasets: {len(intermediate_data['datasets'])} out of {len(metrics_data)}")
    # Save the intermediate data to JSON.
    with open(intermediate_json_filepath, 'w') as f:
        json.dump(intermediate_data, f, indent=4)
    print(f"Intermediate data saved to {intermediate_json_filepath}")

# Now extract data arrays for plotting.
x_vals_anti = []
x_vals_logit = []
y_vals = []
processed_datasets = []

for entry in intermediate_data["datasets"]:
    x_vals_anti.append(entry["anti_steerable_percentage"])
    x_vals_logit.append(entry["mean_logit_diff"])
    y_vals.append(entry["layer13_similarity"])
    processed_datasets.append(entry["dataset"])

# Convert lists to numpy arrays.
x_vals_anti = np.array(x_vals_anti)
x_vals_logit = np.array(x_vals_logit)
y_vals = np.array(y_vals)

# Create plots with two subplots.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Plot 1: Cosine Similarity vs Fraction Anti-Steerable (%)
correlation_anti, p_value_anti = spearmanr(x_vals_anti, y_vals)
ax1.scatter(x_vals_anti, y_vals, color='black', marker='o',
            label=f'Spearman r: {correlation_anti:.3f}\np-value: {p_value_anti:.3e}')
ax1.set_xlabel('Fraction Anti-Steerable (%)')
ax1.set_ylabel('Layer 13 Average Cosine Similarity')
ax1.set_title('Cosine Similarity vs Fraction Anti-Steerable')
ax1.legend(loc='lower left')
ax1.grid(True)

# Plot 2: Cosine Similarity vs Mean Logit Difference
correlation_logit, p_value_logit = spearmanr(x_vals_logit, y_vals)
ax2.scatter(x_vals_logit, y_vals, color='black', marker='s',
            label=f'Spearman r: {correlation_logit:.3f}\np-value: {p_value_logit:.3e}')
ax2.set_xlabel('Mean Logit Difference')
ax2.set_ylabel('Layer 13 Average Cosine Similarity')
ax2.set_title('Cosine Similarity vs Mean Logit Difference')
ax2.legend(loc='lower left')
ax2.grid(True)

plt.tight_layout()
plot_file = os.path.join(plots_path, f'steerability_and_logit_correlation_figure_13_{NUM_SAMPLES}.pdf')
plt.savefig(plot_file)
plt.close()

print(f"Plots saved to {plot_file}")
