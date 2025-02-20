import os
import logging
import time
import argparse
from typing import Dict, List, Any
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
import torch
import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa
import src.utils._validate_custom_datatypes as vcd
from src.utils._validate_custom_datatypes import (
    ActivationListDict,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExperimentConfig:
    def __init__(self, compute_num_samples: int, load_num_samples: int, activation_types: List[int], 
                 device: torch.device):
        """
        Initializes the configuration for model processing.

        Args:
            batch_size (int): Number of sentence pairs to process in a batch.
            num_samples (int): Number of samples to process.
            device (torch.device): The device to run the model on.
        """
        self.compute_num_samples = compute_num_samples
        self.load_num_samples = load_num_samples
        self.activation_types = activation_types
        self.device = device

def load_activations(base_file_path: str, activation_type: str, config: Any) -> Dict[str, ActivationListDict]:
    """
    Load activations from a file and truncate to the specified number of samples.

    Args:
        base_file_path (str): The base file path for the activations.
        activation_type (str): The type of activations to load.

    Returns:
        dict: A dictionary containing the loaded activations.
    """

    file_path = f"{base_file_path}_{activation_type}.pkl"
    logging.info(f"Loading {config.load_num_samples} samples of {activation_type} activations from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # Truncate the loaded data to the specified number of samples
    for key in data:
        vcd._validate_activation_list_dict(data[key])
        for layer in data[key]:
            data[key][layer] = data[key][layer][:config.load_num_samples]
    return data

def plot_distribution(data: Dict[str, List[float]], ax: plt.Axes, title: str, x_label: str,
                      y_label: str, plot_type: str) -> List[str]:
    logging.info(f"Plotting distribution: {title}")
    
    layers = list(data.keys())
    num_layers = len(layers)
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    all_values = []
    layer_means = []
    for i, layer in enumerate(layers):
        values = data[layer]
        all_values.extend(values)
        layer_mean = np.mean(values)
        layer_means.append(layer_mean)
        kde = gaussian_kde(values, bw_method=0.1)
        x_range = np.linspace(min(values), max(values), 200)
        ax.plot(x_range, kde(x_range), color=colors[i], label=f'Layer {layer}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    overall_mean = np.mean(all_values)
    overall_std = np.std(all_values)
    max_mean = max(layer_means)
    max_layer = layers[layer_means.index(max_mean)]
    
    stats_text = f'Mean: {overall_mean:.3f}\n'
    stats_text += f'Std: {overall_std:.3f}\n'
    stats_text += f'Max: {max_mean:.3f}\n'
    stats_text += f'Layer: {max_layer}'
    
    if plot_type == 'cosine':
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:  # norm
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    logging.info(f"Distribution plotting complete: {title}")
    return colors, layers

def main(load_num_samples: int, compute_num_samples: int) -> None:
    start_time = time.time()
    logging.info(f"Starting main function with load_num_samples={load_num_samples}, compute_num_samples={compute_num_samples}")
    
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    DISK_PATH = os.getenv('DISK_PATH')
    if not DISK_PATH:
        raise ValueError("DISK_PATH not found in .env file")
    
    activations_path = os.path.join(DISK_PATH, "contrastive_free_form_activations")
    plots_path = os.path.join(DATA_PATH, "contrastive_free_form_data", "plots")


    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    activation_types = ['penultimate', 'last', 'average']

    config = ExperimentConfig(num_samples=compute_num_samples, load_num_samples=load_num_samples,
                              activation_types=activation_types,
                              device=device)

    models = ['gemma-2-2b-it', 'qwen25_3b_it']# , 'llama32_3b_it', 'llama2_7b_chat']
    # models = ['qwen25_05b_it', 'qwen25_2b_it', 'qwen25_3b_it', 'qwen25_7b_it']
    datasets = [
        "truthfulness_306_en",
        # "sentiment_500_en",
        # "random_500",
        # "toxicity_500_en",
        # "active_to_passive_580_en",
    ]


    logging.info(f"Creating figures for {len(models)} models, {len(datasets)} datasets, and {len(activation_types)} activation types")
    fig_width = 7 * len(models)
    num_cosine_plots = len(datasets) * len(activation_types)
    fig_cos, axs_cos = plt.subplots(num_cosine_plots * 2, len(models), figsize=(fig_width, 10 * num_cosine_plots))
    fig_norm, axs_norm = plt.subplots(num_cosine_plots, len(models), figsize=(fig_width, 5 * num_cosine_plots))

    for model_idx, model in enumerate(models):
        logging.info(f"Processing model: {model} ({model_idx + 1}/{len(models)})")

        # Add model name as column header
        fig_cos.text(model_idx / len(models) + 0.5 / len(models), 0.99, model, 
                     ha='center', va='top', fontsize=14, fontweight='bold')
        fig_norm.text(model_idx / len(models) + 0.5 / len(models), 0.99, model, 
                      ha='center', va='top', fontsize=14, fontweight='bold')

        for dataset_idx, dataset in enumerate(datasets):
            base_file_name = f"{model}_activations_{dataset}_for_{load_num_samples}_samples"
            base_file_path = os.path.join(activations_path, dataset, base_file_name)
            
            for act_idx, act_type in enumerate(activation_types):
                plot_idx = (dataset_idx * len(activation_types) + act_idx)
                
                activations_list_dict = load_activations(base_file_path, act_type, config=config)
                contrastive_activations_list_dict = sv.compute_contrastive_activations(
                    positive_activations=activations_list_dict["positive"], 
                    negative_activations=activations_list_dict["negative"],
                    device=config.device)
                
                steering_vectors = sv.compute_contrastive_steering_vector_dict(
                    positive_activations=activations_list_dict['positive'],
                    negative_activations=activations_list_dict['negative'],
                    device=config.device)

                cosine_sims = cpa.compute_pairwise_cosine_similarity_dict(contrastive_activations_list_dict, config)
                steering_cosine_sims = cpa.compute_many_against_one_cosine_similarity_dict(
                    contrastive_activations_list_dict, steering_vectors, config)
                norm_dist = cpa.compute_l2_norms_dict(contrastive_activations_list_dict, config.device)

                base_title = f"dataset={dataset}\ntoken activation positions: {act_type}"

                ax_cos_pairs = axs_cos[plot_idx * 2, model_idx]
                colors, layers = plot_distribution(cosine_sims, ax_cos_pairs, base_title + "\nPairwise Comparison", 'Cosine Similarity', 'Density', 'cosine')
                ax_cos_pairs.set_xlim(-1, 1)

                ax_cos_steering = axs_cos[plot_idx * 2 + 1, model_idx]
                colors, layers = plot_distribution(steering_cosine_sims, ax_cos_steering, base_title + "\nSteering Vector Comparison", 'Cosine Similarity', 'Density', 'cosine')
                ax_cos_steering.set_xlim(-1, 1)
                
                for ax in [ax_cos_pairs, ax_cos_steering]:
                    sm_cos = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=len(layers)-1))
                    sm_cos.set_array([])
                    cbar_cos = plt.colorbar(sm_cos, ax=ax, label='Layer')
                    tick_locs = range(len(layers))  # Use range for tick locations
                    cbar_cos.set_ticks(tick_locs)
                    cbar_cos.set_ticklabels(layers)

                ax_norm = axs_norm[plot_idx, model_idx]
                colors, layers = plot_distribution(norm_dist, ax_norm, base_title, 'Norm', 'Density', 'norm')

                sm_norm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=len(layers)-1))
                sm_norm.set_array([])
                cbar_norm = plt.colorbar(sm_norm, ax=ax_norm, label='Layer')
                tick_locs = range(len(layers))  # Use range for tick locations
                cbar_norm.set_ticks(tick_locs)
                cbar_norm.set_ticklabels(layers)


    fig_cos.suptitle(f'Distribution of Cosine Similarity of Contrastive Activations for {compute_num_samples} samples', fontsize=16, y=0.995)
    fig_norm.suptitle(f'Distribution of Norms of Contrastive Activations for {compute_num_samples} samples', fontsize=16, y=0.995)
    
    fig_cos.subplots_adjust(top=0.97, hspace=0.5)
    fig_norm.subplots_adjust(top=0.97, hspace=0.5)

    os.makedirs(plots_path, exist_ok=True)

    cos_output_file = os.path.join(plots_path, f'cosine_similarity_distribution_contrastive_activations_{compute_num_samples}_size.pdf')
    fig_cos.savefig(cos_output_file, bbox_inches='tight')
    logging.info(f"Cosine similarity plot saved to: {cos_output_file}")

    norm_output_file = os.path.join(plots_path, f'norm_distribution_contrastive_activations_{compute_num_samples}_size.pdf')
    fig_norm.savefig(norm_output_file, bbox_inches='tight')
    logging.info(f"Norm distribution plot saved to: {norm_output_file}")

    plt.close('all')
    end_time = time.time()
    logging.info(f"Script completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot activation distributions with sample size control.")
    parser.add_argument("--load_samples", type=int, default=100, help="Number of samples to load from each file")
    parser.add_argument("--compute_samples", type=int, default=5, help="Number of samples to use in computations")
    args = parser.parse_args()

    main(args.load_samples, args.compute_samples)