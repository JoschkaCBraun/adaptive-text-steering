import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
import src.utils as utils
import src.utils.compute_steering_vectors as sv

# Constants
NUM_SAMPLES = 500
MODEL_NAME = 'gemma-2-2b-it'
TARGET_LAYER = 20
DATASET_NAME = "sentiment_500_en"

load_dotenv()
device = utils.get_device()

# Paths
file_path = "/Volumes/WD_HDD_MAC/Masters_Thesis_data/contrastive_free_form_activations/sentiment_500_en/gemma-2-2b-it_activations_sentiment_500_en_for_500_samples_last.pkl"
plots_path = os.path.join(utils.get_path('PROJECT_PATH'), "data", "reliability_paper_data", "plots")

def load_activations(file_path: str, num_samples: int, device: torch.device) -> dict:
    """Load and prepare activations for kappa calculation."""
    print(f"Loading activations from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    # Convert loaded data into the expected format
    result = {
        'positive': {},
        'negative': {}
    }
    
    for class_type in ['positive', 'negative']:
        for layer in data[class_type].keys():
            # Convert to list of tensors
            layer_tensors = [tensor.to(device) for tensor in data[class_type][layer][:num_samples]]
            result[class_type][int(layer)] = layer_tensors
            
    return result

# Load activations
activations = load_activations(file_path, NUM_SAMPLES, device)

# Calculate kappa values for both positive and negative samples
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

# Convert to numpy arrays
pos_kappa = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
neg_kappa = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()

# Calculate means
pos_mean = np.mean(pos_kappa)
neg_mean = np.mean(neg_kappa)

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: 2D Scatter
ax1.scatter(pos_kappa, pos_kappa, c='blue', alpha=0.6, label='Positive Samples')
ax1.scatter(neg_kappa, neg_kappa, c='red', alpha=0.6, label='Negative Samples')

# Plot means with larger markers
ax1.scatter(pos_mean, pos_mean, c='blue', s=400, marker='*', label='Positive Mean',
           edgecolor='black', linewidth=2)
ax1.scatter(neg_mean, neg_mean, c='red', s=400, marker='*', label='Negative Mean',
           edgecolor='black', linewidth=2)

# Add the diagonal line
min_val = min(min(pos_kappa), min(neg_kappa))
max_val = max(max(pos_kappa), max(neg_kappa))
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Diagonal')

# Add details to scatter plot
ax1.set_title(f'Kappa Projection of Layer {TARGET_LAYER}\nDataset: {DATASET_NAME}')
ax1.set_xlabel('Kappa Values (x)')
ax1.set_ylabel('Kappa Values (y)')
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.axis('square')

# Plot 2: Histogram
bins = np.linspace(min(min(pos_kappa), min(neg_kappa)),
                  max(max(pos_kappa), max(neg_kappa)), 30)

ax2.hist(neg_kappa, bins=bins, alpha=0.5, color='red', 
         label='Negative', density=False)
ax2.hist(pos_kappa, bins=bins, alpha=0.5, color='blue', 
         label='Positive', density=False)

# Add means as vertical lines
ax2.axvline(x=pos_mean, color='blue', linestyle='--', alpha=0.8)
ax2.axvline(x=neg_mean, color='red', linestyle='--', alpha=0.8)

# Add details to histogram plot
ax2.set_title(f'Kappa Value Distribution\nDataset: {DATASET_NAME}')
ax2.set_xlabel('Kappa Values')
ax2.set_ylabel('Count')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Calculate and display statistics
stats_text = (
    f'Positive Mean: {pos_mean:.3f}\n'
    f'Negative Mean: {neg_mean:.3f}\n'
    f'Positive Std: {np.std(pos_kappa):.3f}\n'
    f'Negative Std: {np.std(neg_kappa):.3f}'
)
ax2.text(0.98, 0.98, stats_text,
         transform=ax2.transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout
plt.tight_layout()

# Save the plot
os.makedirs(plots_path, exist_ok=True)
plt.savefig(os.path.join(plots_path, f'kappa_2d_and_histogram_{DATASET_NAME}_layer_{TARGET_LAYER}.pdf'))
plt.close()