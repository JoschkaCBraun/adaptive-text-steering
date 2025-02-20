import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import src.utils as utils

# Constants
NUM_SAMPLES = 500
MODEL_NAME = 'llama2_7b_chat'
TARGET_LAYER = 13
DATASET_NAME = "corrigible-neutral-HHH"

load_dotenv()
device = utils.get_device()

# Paths
DISK_PATH = utils.get_path('DISK_PATH')
ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
plots_path = os.path.join(utils.get_path('PROJECT_PATH'), "data", "reliability_paper_data", "plots")

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

# Load and process the data
file_name = f"{MODEL_NAME}_activations_{DATASET_NAME}_for_{NUM_SAMPLES}_samples_last.pkl"
file_path = os.path.join(ACTIVATIONS_PATH, DATASET_NAME, file_name)

# Load activations
activations = load_activations(file_path)

# Get layer 13 activations and convert to numpy arrays
layer_key = TARGET_LAYER if TARGET_LAYER in activations['positive'] else str(TARGET_LAYER)
positive_activations = np.array(activations['positive'][layer_key])
negative_activations = np.array(activations['negative'][layer_key])

# Combine activations for PCA
combined_activations = np.vstack([positive_activations, negative_activations])

# Fit PCA
pca = PCA(n_components=2)
transformed_activations = pca.fit_transform(combined_activations)

# Split back into positive and negative
positive_transformed = transformed_activations[:NUM_SAMPLES]
negative_transformed = transformed_activations[NUM_SAMPLES:]

# Create the plot
plt.figure(figsize=(8, 4))

# Plot points
plt.scatter(positive_transformed[:, 0], positive_transformed[:, 1], 
           c='blue', alpha=0.6, label='Positive Samples')
plt.scatter(negative_transformed[:, 0], negative_transformed[:, 1], 
           c='red', alpha=0.6, label='Negative Samples')

# Add details
plt.title(f'PCA Visualization of Layer {TARGET_LAYER} Activations\nDataset: {DATASET_NAME}')
plt.xlabel(f'First Principal Component\n(Explains {pca.explained_variance_ratio_[0]:.1%} of variance)')
plt.ylabel(f'Second Principal Component\n(Explains {pca.explained_variance_ratio_[1]:.1%} of variance)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text box with statistics
total_var = sum(pca.explained_variance_ratio_[:2])
stats_text = f'Total variance explained: {total_var:.1%}'
plt.text(0.02, 0.98, stats_text, 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

# Save the plot
os.makedirs(plots_path, exist_ok=True)
plt.savefig(os.path.join(plots_path, f'pca_visualization_{DATASET_NAME}_layer_{TARGET_LAYER}.pdf'))
plt.close()