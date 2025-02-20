import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import torch
import logging
import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_l2_distributions(original_norms: List[float], corrected_norms: List[float], 
                         scaled_norms: List[float], ax: plt.Axes, title: str) -> None:
    """
    Plot histograms of L2 norms before and after steering vector corrections.
    
    Args:
        original_norms: List of original L2 norms
        corrected_norms: List of L2 norms after steering vector correction
        scaled_norms: List of L2 norms after scaled steering vector correction
        ax: Matplotlib axes to plot on
        title: Title for the plot
    """
    # Calculate means
    orig_mean = np.mean(original_norms)
    corr_mean = np.mean(corrected_norms)
    scaled_mean = np.mean(scaled_norms)
    
    # Create histograms
    bins = np.linspace(
        0,
        max(max(original_norms), max(corrected_norms), max(scaled_norms)),
        30
    )
    
    ax.hist(original_norms, bins=bins, alpha=0.5, color='blue', 
            density=True, label='Original')
    ax.hist(corrected_norms, bins=bins, alpha=0.5, color='red', 
            density=True, label='After Steering')
    ax.hist(scaled_norms, bins=bins, alpha=0.5, color='green', 
            density=True, label='Scaled Steering')
    
    # Add means text box
    stats_text = (
        f'Original μ={orig_mean:.3f}\n'
        f'Corrected μ={corr_mean:.3f}\n'
        f'Scaled μ={scaled_mean:.3f}'
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(title)
    ax.legend()

def compute_difference_dict(negatives: Dict, positives: Dict) -> Dict:
    """
    Compute element-wise differences between negative and positive activations.
    
    Args:
        negatives: Dictionary of negative activations
        positives: Dictionary of positive activations
    
    Returns:
        Dictionary of difference tensors
    """
    return {
        layer: [neg - pos for neg, pos in zip(negatives[layer], positives[layer])]
        for layer in negatives.keys()
    }

def main() -> None:
    # Get paths and device setup
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    DISK_PATH = os.getenv('DISK_PATH')
    if not DISK_PATH:
        raise ValueError("DISK_PATH not found in .env file")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configuration
    model = 'gemma-2-2b-it'
    dataset = 'sentiment_500_en'
    num_samples = 500
    activation_types = ['penultimate', 'last', 'average']
    
    # Load activations
    activations_path = os.path.join(DISK_PATH, "contrastive_free_form_activations")
    base_file_name = f"{model}_activations_{dataset}_for_{num_samples}_samples"
    base_file_path = os.path.join(activations_path, dataset, base_file_name)
    
    # Get all layers from one activation file to determine subplot layout
    with open(f"{base_file_path}_penultimate.pkl", 'rb') as f:
        data = pickle.load(f)
        layers = sorted(list(data['positive'].keys()), key=int)
    
    # Create figure
    fig, axes = plt.subplots(len(layers), len(activation_types), 
                            figsize=(15, 4 * len(layers)))
    
    # Process each activation type
    for act_idx, act_type in enumerate(activation_types):
        logging.info(f"Processing {act_type} activations")
        
        # Load activations
        with open(f"{base_file_path}_{act_type}.pkl", 'rb') as f:
            activations = pickle.load(f)
        
        # Compute steering vector using all samples
        steering_vectors = sv.compute_contrastive_steering_vector_dict(
            positive_activations=activations['positive'],
            negative_activations=activations['negative'],
            device=device
        )
        
        # Process each layer
        for layer_idx, layer in enumerate(layers):
            logging.info(f"Processing layer {layer}")
            
            # Compute original differences and their L2 norms
            original_differences = compute_difference_dict(
                activations['negative'],
                activations['positive']
            )
            original_l2_norms = cpa.compute_l2_norms_dict(original_differences, device)
            
            # Compute corrected differences and their L2 norms
            # First add steering vector to negative activations
            corrected_negatives = {
                layer: [neg + steering_vectors[layer] 
                       for neg in activations['negative'][layer]]
            }
            # Then compute differences with positives
            corrected_differences = compute_difference_dict(
                corrected_negatives,
                activations['positive']
            )
            corrected_l2_norms = cpa.compute_l2_norms_dict(corrected_differences, device)
            
            scale_factors = sv.calculate_feature_expressions_kappa(
                activations['negative'][layer],
                activations['negative'][layer],
                activations['positive'][layer],
                device=device
            )

            # Apply scaled steering vector
            scaled_negatives = {
                layer: [neg + scale_factor * steering_vector for neg, scale_factor, steering_vector in zip(
                    activations['negative'][layer], scale_factors, steering_vectors[layer])]
            }

            # Compute differences with scaled steering vector
            scaled_differences = compute_difference_dict(
                scaled_negatives,
                activations['positive']
            )
            scaled_l2_norms = cpa.compute_l2_norms_dict(scaled_differences, device)

            if len(layers) == 1:
                current_ax = axes[act_idx]
            else:
                current_ax = axes[layer_idx, act_idx]

            # Update plot call to include scaled norms
            plot_l2_distributions(
                original_l2_norms[layer],
                corrected_l2_norms[layer],
                scaled_l2_norms[layer],
                current_ax,
                f"{act_type} - Layer {layer}"
            )
    
    # Adjust layout and save
    plt.tight_layout()
    plots_path = os.path.join(DATA_PATH, "contrastive_free_form_data", "plots")
    os.makedirs(plots_path, exist_ok=True)
    output_file = os.path.join(
        plots_path, 
        f'l2_error_distribution_before_after_steering_{model}_{dataset}_{num_samples}.pdf'
    )
    plt.savefig(output_file, bbox_inches='tight')
    logging.info(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()