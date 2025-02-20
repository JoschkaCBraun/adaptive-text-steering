import os
import pickle
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from dotenv import load_dotenv
import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa
import src.utils as utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Check if mps device is available
device = utils.get_device()
logging.info(f"Using device: {device}")

def load_activations(file_path: str, load_num_samples: int) -> dict:
    """Load and preprocess activations from a pickle file."""
    logging.info(f"Loading {load_num_samples} samples of activations from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert structure to match ActivationListDict format
    processed_data = {
        'positive': {
            int(layer): list(data['positive'][layer][:load_num_samples])
            for layer in data['positive'].keys()
        },
        'negative': {
            int(layer): list(data['negative'][layer][:load_num_samples])
            for layer in data['negative'].keys()
        }
    }
    
    return processed_data

def plot_distribution(data: dict, ax: plt.Axes, title: str, x_label: str, y_label: str, plot_type: str) -> tuple:
    logging.info(f"Plotting distribution: {title}")
    layers = sorted(list(data.keys()))
    num_layers = len(layers)
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    all_values = []
    for i, layer in enumerate(layers):
        values = data[layer]
        all_values.extend(values)
        ax.hist(values, bins=30, density=True, alpha=0.7, color=colors[i], 
               label=f'Layer {layer}', histtype='step', linewidth=3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    mean = np.mean(all_values)
    std = np.std(all_values)
    stats_text = f'Mean: {mean:.3f}\nStd: {std:.3f}'
    
    text_pos = (0.02, 0.98) if plot_type == 'cosine' else (0.98, 0.98)
    text_align = 'left' if plot_type == 'cosine' else 'right'
    
    ax.text(*text_pos, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment=text_align,
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    logging.info(f"Distribution plotting complete: {title}")
    return colors, layers

def get_colorbar_ticks(num_layers: int) -> list:
    if num_layers <= 6:
        return list(range(1, num_layers + 1))
    elif num_layers <= 10:
        return [1, num_layers // 2, num_layers]
    else:
        return [1, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers]

def main() -> None:
    compute_num_samples = 100
    load_num_samples = 500
    if compute_num_samples > load_num_samples:
        raise ValueError("compute_num_samples must be less than or equal to load_num_samples")
    start_time = time.time()
    logging.info(f"Starting main function with load_num_samples={load_num_samples}, compute_num_samples={compute_num_samples}")
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    DISK_PATH = os.getenv('DISK_PATH')
    if not DISK_PATH:
        raise ValueError("DISK_PATH not found in .env file")
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")
    
    models = [
        ('gemma-2-2b-it', 'google/gemma-2-2b-it', [0, 5, 10, 13, 20, 25]),
        ('llama-3-3b-it', 'meta-llama/Llama-3.2-3B-Instruct', [0, 5, 10, 13, 20, 25])
        # ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [1, 5, 10, 13, 20]), # 13 indicates the layer
        # ('qwen1.5_14b_chat', 'Qwen/Qwen1.5-14B-Chat', [21]) # 21 indicates the layer
    ]

    datasets = [
        "anti-LGBTQ-rights",
        "believes-AIs-are-not-an-existential-threat-to-humanity",
        # "believes-it-is-not-being-watched-by-humans",
        # "narcissism",
        # "openness",
        # "subscribes-to-average-utilitarianism",
        # "subscribes-to-deontology",
        # "willingness-to-use-physical-force-to-achieve-benevolent-goals"
    ]
    activation_types = ['last']

    n_rows_cos = len(datasets) * len(activation_types) * 2  # Each dataset+activation type needs 2 rows for cosine plots
    n_rows_norm = len(datasets) * len(activation_types)     # Each dataset+activation type needs 1 row for norm plots

    logging.info(f"Creating figures for {len(models)} models, {len(datasets)} datasets, and {len(activation_types)} activation types")
    fig_cos, axs_cos = plt.subplots(n_rows_cos, len(models), figsize=(20, 7 * n_rows_cos))
    fig_norm, axs_norm = plt.subplots(n_rows_norm, len(models), figsize=(20, 7 * n_rows_norm))


    for model_idx, model in enumerate(models):
        model_name, _, model_layer = model
        logging.info(f"Processing model: {model_name} ({model_idx + 1}/{len(models)})")
        for plot_idx, (dataset, act_type) in enumerate([(d, a) for d in datasets for a in activation_types]):
            logging.info(f"Processing dataset: {dataset}, activation type: {act_type}")
            file_name = f"{model_name}_activations_{dataset}_for_{load_num_samples}_samples_{act_type}.pkl"
            file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
            
            activations = load_activations(file_path, load_num_samples)
            contrastive_activations = sv.compute_contrastive_activations(activations['positive'], activations['negative'], device)
            layers = list(activations['positive'].keys())
            steering_vectors = sv.compute_contrastive_steering_vector_dict(activations['positive'], activations['negative'], device)
            cosine_sims = cpa.compute_pairwise_cosine_similarity_dict(contrastive_activations, device)
            steering_cosine_sims = cpa.compute_many_against_one_cosine_similarity_dict(contrastive_activations, steering_vectors, device)
            norm_dist = cpa.compute_l2_norms_dict(contrastive_activations, device)

            base_title = f"model: {model_name}, lang={dataset}\ntoken activations: {act_type}, samples={compute_num_samples}"


            logging.info("Plotting cosine similarity (pairwise)")
            ax_cos_pairs = axs_cos[plot_idx * 2, model_idx]
            colors, num_layers = plot_distribution(cosine_sims, ax_cos_pairs, base_title + "\nPairwise Comparison", 'Cosine Similarity', 'Density', 'cosine')
            ax_cos_pairs.set_xlim(-1, 1)

            logging.info("Plotting cosine similarity (steering vector)")
            ax_cos_steering = axs_cos[plot_idx * 2 + 1, model_idx]
            colors, num_layers = plot_distribution(steering_cosine_sims, ax_cos_steering, base_title + "\nSteering Vector Comparison", 'Cosine Similarity', 'Density', 'cosine')
            ax_cos_steering.set_xlim(-1, 1)

            logging.info("Adding colorbars to cosine similarity plots")
            for ax in [ax_cos_pairs, ax_cos_steering]:
                sm_cos = ScalarMappable(cmap='viridis', norm=Normalize(vmin=min(layers), vmax=max(layers)))
                sm_cos.set_array([])
                cbar_cos = plt.colorbar(sm_cos, ax=ax, label='Layer')
                cbar_cos.set_ticks(layers)
                cbar_cos.set_ticklabels(layers)

            logging.info("Plotting norm distribution")
            ax_norm = axs_norm[plot_idx, model_idx]
            colors, num_layers = plot_distribution(norm_dist, ax_norm, base_title, 'Norm', 'Density', 'norm')
            
            logging.info("Adding colorbar to norm distribution plot")
            sm_norm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=min(layers), vmax=max(layers)))
            sm_norm.set_array([])
            cbar_norm = plt.colorbar(sm_norm, ax=ax_norm, label='Layer')
            cbar_norm.set_ticks(layers)
            cbar_norm.set_ticklabels(layers)

    logging.info("Adding main titles to figures")
    fig_cos.suptitle('Distribution of Cosine Similarity of Contrastive Activations', fontsize=16, y=0.995)
    fig_norm.suptitle('Distribution of Norms of Contrastive Activations', fontsize=16, y=0.995)

    logging.info("Adjusting figure spacing")
    fig_cos.subplots_adjust(top=0.97, hspace=0.5)
    fig_norm.subplots_adjust(top=0.97, hspace=0.5)

    logging.info("Creating plots directory")
    os.makedirs(plots_path, exist_ok=True)

    logging.info("Saving figures")
    cos_output_file = os.path.join(plots_path, f'cosine_similarity_distribution_contrastive_activations_caa_dataset_{compute_num_samples}.pdf')
    fig_cos.savefig(cos_output_file, bbox_inches='tight') # dpi=300 for high resolution
    logging.info(f"Cosine similarity plot saved to: {cos_output_file}")

    norm_output_file = os.path.join(plots_path, f'norm_distribution_contrastive_activations_caa_dataset_{compute_num_samples}.pdf')
    fig_norm.savefig(norm_output_file, bbox_inches='tight') # dpi=300 for high resolution
    logging.info(f"Norm distribution plot saved to: {norm_output_file}")

    plt.close('all')

    end_time = time.time()
    logging.info(f"Script completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()