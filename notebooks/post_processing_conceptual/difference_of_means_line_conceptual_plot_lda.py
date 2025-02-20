import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Combine activations and create labels
X = np.vstack([positive_activations, negative_activations])
y = np.array([1] * NUM_SAMPLES + [0] * NUM_SAMPLES)

# Fit LDA with one component
lda = LinearDiscriminantAnalysis(n_components=1)
transformed_activations = lda.fit_transform(X, y)

# Split back into positive and negative
positive_transformed = transformed_activations[:NUM_SAMPLES]
negative_transformed = transformed_activations[NUM_SAMPLES:]

# Create the plot
plt.figure(figsize=(12, 6))

# Create histograms for both classes
plt.hist(negative_transformed, bins=30, alpha=0.5, color='red', label='Negative Samples')
plt.hist(positive_transformed, bins=30, alpha=0.5, color='blue', label='Positive Samples')

# Add details
plt.title(f'LDA Projection of Layer {TARGET_LAYER} Activations\nDataset: {DATASET_NAME}')
plt.xlabel('Discriminant Component')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)

# Calculate and display classification accuracy
accuracy = lda.score(X, y)
stats_text = f'LDA Classification Accuracy: {accuracy:.1%}'
plt.text(0.02, 0.98, stats_text, 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

# Print additional statistics
print(f"\nLDA Classification Accuracy: {accuracy:.3f}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_[0]:.3f}")

# Save the plot
os.makedirs(plots_path, exist_ok=True)
plt.savefig(os.path.join(plots_path, f'lda_visualization_1d_{DATASET_NAME}_layer_{TARGET_LAYER}.pdf'))
plt.close()