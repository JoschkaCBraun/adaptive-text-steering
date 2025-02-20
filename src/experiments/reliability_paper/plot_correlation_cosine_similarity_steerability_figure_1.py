import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from dotenv import load_dotenv
import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa
import src.utils as utils
import src.utils.dataset_names as d

NUM_SAMPLES = 500
MODEL_NAME = 'llama2_7b_chat'
TARGET_LAYER = 13

load_dotenv()
device = utils.get_device()

datasets = d.all_datasets_with_fraction_anti_steerable_figure_1_tan_et_al
dataset_to_score = {name: score for name, _, score in datasets}
dataset_to_index = {name: index for name, index, _ in datasets}

PROJECT_PATH = utils.get_path('PROJECT_PATH')
DISK_PATH = utils.get_path('DISK_PATH')
ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")

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


# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x_vals_score = []
x_vals_index = []
y_vals = []
processed_datasets = []

for dataset in dataset_to_score.keys():
    try:
        similarity = compute_layer_13_similarity(dataset)
        x_vals_score.append(dataset_to_score[dataset])
        x_vals_index.append(dataset_to_index[dataset])
        y_vals.append(similarity)
        processed_datasets.append(dataset)
        print(f"Successfully processed {dataset}, similarity: {similarity:.3f}")
    except Exception as e:
        print(f"Error processing dataset {dataset}")
        print(f"Error details: {str(e)}")
        print(f"Error type: {type(e)}")

print(f"\nSuccessfully processed datasets: {processed_datasets}")
print(f"Total processed: {len(processed_datasets)} out of {len(dataset_to_score)}")

x_vals_score = np.array(x_vals_score)
x_vals_index = np.array(x_vals_index)
y_vals = np.array(y_vals)

# Plot 1: Fraction Anti-Steerable vs Layer 13 Cosine Similarity
correlation_score, p_value_score = pearsonr(x_vals_score, y_vals)
ax1.scatter(x_vals_score, y_vals, color='blue', marker='o',
           label=f'r={correlation_score:.3f}, p={p_value_score:.3f}')
ax1.set_xlabel('Fraction Anti-Steerable')
ax1.set_ylabel('Layer 13 Steering Vector Cosine Similarity')
ax1.set_title('Steerability vs Layer 13 Cosine Similarity')
ax1.legend()
ax1.grid(True)

# Plot 2: Index vs Layer 13 Cosine Similarity
correlation_index, p_value_index = pearsonr(x_vals_index, y_vals)
ax2.scatter(x_vals_index, y_vals, color='red', marker='s',
           label=f'r={correlation_index:.3f}, p={p_value_index:.3f}')
ax2.set_xlabel('Dataset Index')
ax2.set_ylabel('Layer 13 Steering Vector Cosine Similarity')
ax2.set_title('Dataset Index vs Layer 13 Cosine Similarity')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
os.makedirs(plots_path, exist_ok=True)
plt.savefig(os.path.join(plots_path, f'steerability_and_index_correlation_figure_1_{NUM_SAMPLES}.pdf'))
plt.close()