"""
plot_norm_and_cosine_similarity_distribution.py
"""

import os
import pickle
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
from dotenv import load_dotenv

import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_activations(file_path: str, load_num_samples: int, selected_layers: list) -> dict:
    """Load activations and filter for selected layers."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return {
        'positive': {
            int(layer): list(data['positive'][layer][:load_num_samples])
            for layer in data['positive'].keys()
            if int(layer) in selected_layers
        },
        'negative': {
            int(layer): list(data['negative'][layer][:load_num_samples])
            for layer in data['negative'].keys()
            if int(layer) in selected_layers
        }
    }

def compute_l2_norms(contrastive_activations: dict) -> dict:
    """Compute L2 norms for contrastive activations."""
    norms = {}
    for layer, activations in contrastive_activations.items():
        # Fixed tensor creation to avoid warning
        norms[layer] = [torch.norm(act.clone().detach()).item() if isinstance(act, torch.Tensor)
                       else torch.norm(torch.tensor(act, device=device)).item() 
                       for act in activations]
    return norms

def plot_distribution(data: dict, ax: plt.Axes, title: str, plot_type: str = 'cosine', dataset: str = '') -> None:
    layers = sorted(list(data.keys()))
    num_layers = len(layers)
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    if dataset == "believes-it-is-not-being-watched-by-humans":
        dataset = "believes-not-being-watched"
    all_values = []
    layer_stats = {}
    
    for i, layer in enumerate(layers):
        values = data[layer]
        if plot_type == 'l2':
            # Normalize the L2 norms by their mean
            mean_norm = np.mean(values)
            values = [v/mean_norm for v in values]
        all_values.extend(values)
        layer_stats[layer] = np.mean(values)
        layer_stats[layer] = np.mean(values)
        ax.hist(values, bins=30, density=True, alpha=0.7, color=colors[i], 
               label=f'Layer {layer}', histtype='step', linewidth=3)

    if plot_type == 'cosine':
        mean = np.mean(all_values)
        std = np.std(all_values)
        max_layer = max(layer_stats.items(), key=lambda x: x[1])
        stats_text = (f'Avg Mean: {mean:.2f}\n'
                     f'Avg Std: {std:.2f}\n'
                     f'Max Mean: {max_layer[1]:.2f}\n'
                     f'Max Layer: {max_layer[0]}')
        ax.set_xlabel('Cosine Similarity')
        ax.set_xlim(-1, 1)
        # Set new title for cosine plot
        plot_title = f'Cosine Similarity with CAA vector (n=500)\nLlama2 7b chat, {dataset}'
    else:  # L2 norm
        mean = np.mean(all_values)
        std = np.std(all_values)
        min_layer = min((layer, np.std(data[layer])) for layer in layers)
        stats_text = (f'Avg Std: {std:.2f}\n'
                     f'Min Layer: {min_layer[0]}')
        ax.set_xlabel('L2 Norm')
        # Set new title for L2-norm plot
        plot_title = f'Contrastive Activation L2-norm distribution (n=500)\nLlama2 7b chat, {dataset}'
        
    ax.set_ylabel('Density')
    ax.set_title(plot_title)

    if plot_type == 'cosine':
        text_x = 0.02  # left side
    else:  # L2 norm
        text_x = 0.66  # right side
    
    ax.text(text_x, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=min(layers), vmax=max(layers)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Layer')
    cbar.set_ticks(layers)
    cbar.set_ticklabels(layers)



def main() -> None:
    compute_num_samples = 500
    load_num_samples = 500
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    DISK_PATH = os.getenv('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")

    selected_layers = list(range(10, 31, 5))  # layers 10-20
    model = ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', selected_layers)
    datasets = ["corrigible-neutral-HHH", "myopic-reward", "believes-it-is-not-being-watched-by-humans"]
    activation_types = ['last']

    fig, axs = plt.subplots(len(datasets), 2, figsize=(10, 10))
   
    for dataset_idx, dataset in enumerate(datasets):
        for act_type in activation_types:
            file_name = f"{model[0]}_activations_{dataset}_for_{load_num_samples}_samples_{act_type}_all_layers.pkl"
            file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
            
            activations = load_activations(file_path, load_num_samples, selected_layers)
            contrastive_activations = sv.compute_contrastive_activations(activations['positive'], activations['negative'], device)
            steering_vectors = sv.compute_contrastive_steering_vector_dict(activations['positive'], activations['negative'], device)
            
            # Compute cosine similarities and L2 norms
            steering_cosine_sims = cpa.compute_many_against_one_cosine_similarity_dict(contrastive_activations, steering_vectors, device)
            l2_norms = compute_l2_norms(contrastive_activations)
            
            # Plot steering vector cosine similarities
            plot_distribution(steering_cosine_sims, axs[dataset_idx, 0], 
                            "", 'cosine', dataset)
            
            # Plot L2 norms
            plot_distribution(l2_norms, axs[dataset_idx, 1], 
                            "", 'l2', dataset)

    fig.suptitle('Distribution Analysis of Contrastive Activations', fontsize=16, y=0.995)
    fig.tight_layout()
    
    os.makedirs(plots_path, exist_ok=True)
    output_file = os.path.join(plots_path, f'activation_distributions_{compute_num_samples}.pdf')
    fig.savefig(output_file, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    main()