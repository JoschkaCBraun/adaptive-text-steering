import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torch
import logging
from src.utils import compute_steering_vectors as sv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_all_projections(pos_proj: np.ndarray, 
                        neg_proj: np.ndarray,
                        basic_steered_proj: np.ndarray,
                        scaled_steered_proj: np.ndarray,
                        ax: plt.Axes, 
                        title: str) -> None:
    """
    Plot histograms of all projections onto steering vector.
    
    Args:
        pos_proj: Projected positive activations
        neg_proj: Projected negative activations
        basic_steered_proj: Projected basic steered negative activations
        scaled_steered_proj: Projected scaled steered negative activations
        ax: Matplotlib axes to plot on
        title: Title for the plot
    """
    # Create histograms
    bins = np.linspace(
        min(np.min(pos_proj), np.min(neg_proj), 
            np.min(basic_steered_proj), np.min(scaled_steered_proj)),
        max(np.max(pos_proj), np.max(neg_proj), 
            np.max(basic_steered_proj), np.max(scaled_steered_proj)),
        30
    )
    
    ax.hist(neg_proj, bins=bins, alpha=0.5, color='blue', 
            density=True, label='Original Negative')
    ax.hist(pos_proj, bins=bins, alpha=0.5, color='red', 
            density=True, label='Original Positive')
    ax.hist(basic_steered_proj, bins=bins, alpha=0.5, color='green', 
            density=True, label='Basic Steered')
    ax.hist(scaled_steered_proj, bins=bins, alpha=0.5, color='purple', 
            density=True, label='Scaled Steered')
    
    # Add means text box
    stats_text = (
        f'Pos μ={np.mean(pos_proj):.3f}\n'
        f'Neg μ={np.mean(neg_proj):.3f}\n'
        f'Basic μ={np.mean(basic_steered_proj):.3f}\n'
        f'Scaled μ={np.mean(scaled_steered_proj):.3f}'
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(title)
    ax.legend()

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
            
            # Get steering vector for this layer
            steering_vector = steering_vectors[layer]
            
            # Project original activations
            pos_proj = sv.calculate_feature_expressions_kappa(
                activations['positive'][layer],
                activations['positive'][layer],
                activations['negative'][layer],
                device=device
            )
            neg_proj = sv.calculate_feature_expressions_kappa(
                activations['negative'][layer],
                activations['positive'][layer],
                activations['negative'][layer],
                device=device
            )
            
            # Project basic steered activations
            basic_steered = [neg + steering_vector 
                            for neg in activations['negative'][layer]]
            basic_steered_proj = sv.calculate_feature_expressions_kappa(
                basic_steered,
                activations['positive'][layer],
                activations['negative'][layer],
                device=device
            )
            
            scale_factors = sv.calculate_feature_expressions_kappa(
                activations['negative'][layer], activations['negative'][layer],
                activations['positive'][layer], device=device
            )

            # Apply scaled steering vector
            scaled_steered = [neg + scale_factor * steering_vector for neg, scale_factor, steering_vector in zip(
                activations['negative'][layer], scale_factors, steering_vectors[layer])]

            scaled_steered_proj = sv.calculate_feature_expressions_kappa(
                scaled_steered,
                activations['positive'][layer],
                activations['negative'][layer],
                device=device
            )
            
            # Get the correct axes based on number of layers
            if len(layers) == 1:
                current_ax = axes[act_idx]
            else:
                current_ax = axes[layer_idx, act_idx]

            # transform pos_proj, neg_proj, basic_steered_proj, scaled_steered_proj to numpy arrays
            pos_proj = np.array(pos_proj)
            neg_proj = np.array(neg_proj)
            basic_steered_proj = np.array(basic_steered_proj)
            scaled_steered_proj = np.array(scaled_steered_proj)
            
            # Plot all distributions
            plot_all_projections(
                pos_proj, 
                neg_proj,
                basic_steered_proj,
                scaled_steered_proj,
                current_ax,
                f"{act_type} - Layer {layer}"
            )
    
    # Adjust layout and save
    plt.tight_layout()
    plots_path = os.path.join(DATA_PATH, "contrastive_free_form_data", "plots")
    os.makedirs(plots_path, exist_ok=True)
    output_file = os.path.join(
        plots_path, 
        f'projection_distributions_comparison_{model}_{dataset}_{num_samples}.pdf'
    )
    plt.savefig(output_file, bbox_inches='tight')
    logging.info(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()