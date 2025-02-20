import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import src.utils as utils

# Constants
NUM_SAMPLES = 500
MODEL_NAME = 'gemma-2-2b-it'
TARGET_LAYER = 0
DATASET_NAME = "sentiment_500_en"

load_dotenv()
device = utils.get_device()

# Paths
file_path = "/Volumes/WD_HDD_MAC/Masters_Thesis_data/contrastive_free_form_activations/sentiment_500_en/gemma-2-2b-it_activations_sentiment_500_en_for_500_samples_last.pkl"
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

# Load activations
activations = load_activations(file_path)

# Get layer activations and convert to numpy arrays
layer_key = TARGET_LAYER if TARGET_LAYER in activations['positive'] else str(TARGET_LAYER)
# Move tensors to CPU before converting to numpy
positive_activations = np.array([tensor.cpu().numpy() for tensor in activations['positive'][layer_key]])
negative_activations = np.array([tensor.cpu().numpy() for tensor in activations['negative'][layer_key]])

# Combine activations for PCA
combined_activations = np.vstack([positive_activations, negative_activations])

# Fit PCA
pca = PCA(n_components=2)
transformed_activations = pca.fit_transform(combined_activations)

# Split back into positive and negative
positive_transformed = transformed_activations[:NUM_SAMPLES]
negative_transformed = transformed_activations[NUM_SAMPLES:]

# Calculate means
positive_mean = np.mean(positive_transformed, axis=0)
negative_mean = np.mean(negative_transformed, axis=0)

# Create the plot
plt.figure(figsize=(10, 5))

# Plot points
plt.scatter(positive_transformed[:, 0], positive_transformed[:, 1], 
           c='blue', alpha=0.6, label='Positive Samples')
plt.scatter(negative_transformed[:, 0], negative_transformed[:, 1], 
           c='red', alpha=0.6, label='Negative Samples')

# Calculate the direction vector between means
direction = negative_mean - positive_mean
# Get the current axis limits
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

# Calculate line endpoints that extend to the plot boundaries
# Parametric line equation: point = mean + t * direction
# Solve for t at x bounds and y bounds
tx = [(xlim[i] - positive_mean[0]) / direction[0] for i in range(2)]
ty = [(ylim[i] - positive_mean[1]) / direction[1] for i in range(2)]
# Combine all valid intersection points
t_values = [t for t in tx + ty if not np.isinf(t)]
t_min, t_max = min(t_values), max(t_values)

# Calculate endpoints
start_point = positive_mean + t_min * direction * 0.4
end_point = positive_mean + t_max * direction * 0.5

# Plot the connecting line
plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
         'k--', alpha=1, label='Difference of Means Line', linewidth=2)

# Plot means with larger markers
plt.scatter(positive_mean[0], positive_mean[1], 
           c='blue', s=400, marker='*', label='Positive Mean',
           edgecolor='black', linewidth=2)
plt.scatter(negative_mean[0], negative_mean[1], 
           c='red', s=400, marker='*', label='Negative Mean',
           edgecolor='black', linewidth=2)

# Add details
plt.title(f'PCA Visualization of Layer {TARGET_LAYER} Activations of Sentiment Dataset')
plt.xlabel(f'First Principal Component (Explains {pca.explained_variance_ratio_[0]:.1%} of variance)')
plt.ylabel(f'Second Principal Component (Explains {pca.explained_variance_ratio_[1]:.1%} of variance)')
# increase font size
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)

# Save the plot
os.makedirs(plots_path, exist_ok=True)
plt.savefig(os.path.join(plots_path, f'pca_visualization_{DATASET_NAME}_layer_{TARGET_LAYER}.pdf'))
plt.close()