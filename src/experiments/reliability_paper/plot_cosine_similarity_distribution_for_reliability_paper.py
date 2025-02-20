"""
plot_cosine_similarity_distribution_for_reliability_paper.py
"""
import os
import json
import pickle
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict
import numpy as np
import torch
from dotenv import load_dotenv
import src.utils.compute_steering_vectors as sv
import src.utils.compute_properties_of_activations as cpa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def load_activations(file_path: str, load_num_samples: int) -> dict:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return {
        'positive': {
            int(layer): list(data['positive'][layer][:load_num_samples])
            for layer in data['positive'].keys()
        },
        'negative': {
            int(layer): list(data['negative'][layer][:load_num_samples])
            for layer in data['negative'].keys()
        }
    }

def plot_distribution(data: dict, ax: plt.Axes, title: str) -> tuple:
    layers = sorted(list(data.keys()))
    num_layers = len(layers)
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    all_values = []
    layer_means = {}
    for i, layer in enumerate(layers):
        values = data[layer]
        all_values.extend(values)
        layer_means[layer] = np.mean(values)
        ax.hist(values, bins=30, density=True, alpha=0.7, color=colors[i], 
               label=f'Layer {layer}', histtype='step', linewidth=3)

    # Calculate statistics
    mean = np.mean(all_values)
    std = np.std(all_values)
    max_layer = max(layer_means.items(), key=lambda x: x[1])
    
    stats_text = (f'Mean: {mean:.3f}\n'
                 f'Std: {std:.3f}\n'
                 f'Max Mean: {max_layer[1]:.3f}\n'
                 f'Max Layer: {max_layer[0]}')
    
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return colors, layers

def extract_legend_stats(data: dict) -> Dict[str, dict]:
    all_values = []
    layer_means = {}
    
    for layer in data.keys():
        values = data[layer]
        layer_means[layer] = float(np.mean(values))
        all_values.extend(values)
    
    mean = float(np.mean(all_values))
    std = float(np.std(all_values))
    max_layer, max_mean = max(layer_means.items(), key=lambda x: x[1])
    
    return {
        'mean': mean,
        'std': std,
        'max_mean': max_mean,
        'max_layer': int(max_layer),
        'layer_means': {int(k): v for k, v in layer_means.items()}
    }

def main() -> None:
    compute_num_samples = 100
    load_num_samples = 100
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    DISK_PATH = os.getenv('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")
    results_path = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "result_statistics")

    
    models = [
        # ('gemma-2-2b-it', 'google/gemma-2-2b-it', [0, 5, 10, 13, 20, 25]),
        ('llama-3-3b-it', 'meta-llama/Llama-3.2-3B-Instruct', [0, 5, 10, 13, 20, 25]),
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [1, 5, 10, 13, 20]), # 13 indicates the layer

    ]

    persona_datasets = [
        "anti-LGBTQ-rights",
        "believes-AIs-are-not-an-existential-threat-to-humanity",
        "believes-it-is-not-being-watched-by-humans",
        "narcissism",
        "openness",
        "subscribes-to-average-utilitarianism",
        "subscribes-to-deontology",
        "willingness-to-use-physical-force-to-achieve-benevolent-goals"
    ]

    x_risk_datasets = [
        "corrigible-neutral-HHH",
        "myopic-reward",
        "self-awareness-good-text-model",
        "self-awareness-text-model",
        "self-awareness-training-web-gpt",
    ]

    datasets = persona_datasets + x_risk_datasets
    activation_types = ['last']

    n_rows = len(datasets) * len(activation_types) * 2
    fig, axs = plt.subplots(n_rows, len(models), figsize=(20, 7 * n_rows))
    combined_stats = {}

    for model_idx, (model_name, _, _) in enumerate(models):
        combined_stats[model_name] = {}
        for plot_idx, (dataset, act_type) in enumerate([(d, a) for d in datasets for a in activation_types]):
            file_name = f"{model_name}_activations_{dataset}_for_{load_num_samples}_samples_{act_type}.pkl"
            file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
            
            activations = load_activations(file_path, load_num_samples)
            contrastive_activations = sv.compute_contrastive_activations(activations['positive'], activations['negative'], device)
            layers = list(activations['positive'].keys())
            steering_vectors = sv.compute_contrastive_steering_vector_dict(activations['positive'], activations['negative'], device)
            
            cosine_sims = cpa.compute_pairwise_cosine_similarity_dict(contrastive_activations, device)
            steering_cosine_sims = cpa.compute_many_against_one_cosine_similarity_dict(contrastive_activations, steering_vectors, device)

            
            combined_stats[model_name][dataset] = {
                'pairwise': extract_legend_stats(cosine_sims),
                'steering': extract_legend_stats(steering_cosine_sims)
            }
            
            base_title = f"model: {model_name}, lang={dataset}\ntoken activations: {act_type}, samples={compute_num_samples}"
            
            # Pairwise comparison plot
            ax_pairs = axs[plot_idx * 2, model_idx]
            plot_distribution(cosine_sims, ax_pairs, base_title + "\nPairwise Comparison")
            
            # Steering vector comparison plot
            ax_steering = axs[plot_idx * 2 + 1, model_idx]
            plot_distribution(steering_cosine_sims, ax_steering, base_title + "\nSteering Vector Comparison")
            
            # Add colorbars
            for ax in [ax_pairs, ax_steering]:
                sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=min(layers), vmax=max(layers)))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='Layer')
                cbar.set_ticks(layers)
                cbar.set_ticklabels(layers)

    output_path = os.path.join(results_path, "combined_legend_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(combined_stats, f, indent=4)

    fig.suptitle('Distribution of Cosine Similarity of Contrastive Activations', fontsize=16, y=0.995)
    fig.subplots_adjust(top=0.97, hspace=0.5)
    
    os.makedirs(plots_path, exist_ok=True)
    output_file = os.path.join(plots_path, f'cosine_similarity_distribution_contrastive_activations_{compute_num_samples}.pdf')
    fig.savefig(output_file, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()