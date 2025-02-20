import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from dotenv import load_dotenv
import torch
import logging
import time
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Check if mps device is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def load_activations(file_path, load_num_samples):
    logging.info(f"Loading {load_num_samples} samples of activations from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # Truncate the loaded data to the specified number of samples
    for key in data:
        data[key] = data[key][:load_num_samples]
    return data

def compute_steering_vector(activations, compute_num_samples):
    logging.info(f"Computing steering vector using {compute_num_samples} samples")
    steering_vectors = {}
    for layer in activations['positive'][0].keys():
        contrastive_acts = []
        for pos, neg in zip(activations['positive'][:compute_num_samples], activations['negative'][:compute_num_samples]):
            pos_act = pos[layer].to(device)
            neg_act = neg[layer].to(device)
            contrastive_acts.append(pos_act - neg_act)
        steering_vectors[layer] = torch.mean(torch.stack(contrastive_acts), dim=0)
    logging.info("Steering vector computation complete")
    return steering_vectors

def compute_cosine_similarity(activations, compute_num_samples):
    logging.info(f"Computing pairwise cosine similarities using {compute_num_samples} samples")
    cosine_sims = {}
    for layer in activations['positive'][0].keys():
        contrastive_acts = []
        for pos, neg in zip(activations['positive'][:compute_num_samples], activations['negative'][:compute_num_samples]):
            pos_act = pos[layer].to(device)
            neg_act = neg[layer].to(device)
            contrastive_acts.append(pos_act - neg_act)
        
        # Stack all contrastive activations
        contrastive_acts_stacked = torch.stack(contrastive_acts)
        
        # Flatten the activations along the last dimension
        contrastive_acts_flat = contrastive_acts_stacked.view(contrastive_acts_stacked.size(0), -1)
        
        # Compute pairwise cosine similarities
        norm = torch.norm(contrastive_acts_flat, dim=1, keepdim=True)
        normalized = contrastive_acts_flat / norm
        cosine_sim_matrix = torch.mm(normalized, normalized.t())
        
        # Extract upper triangular part (excluding diagonal)
        cosine_sims[layer] = cosine_sim_matrix[torch.triu(torch.ones_like(cosine_sim_matrix), diagonal=1) == 1].cpu().numpy()

    logging.info("Pairwise cosine similarity computation complete")
    return cosine_sims


def compute_steering_cosine_similarity(activations, steering_vectors, compute_num_samples):
    logging.info(f"Computing steering vector cosine similarities using {compute_num_samples} samples")
    steering_cosine_sims = {}
    for layer in activations['positive'][0].keys():
        contrastive_acts = []
        for pos, neg in zip(activations['positive'][:compute_num_samples], activations['negative'][:compute_num_samples]):
            pos_act = pos[layer].to(device)
            neg_act = neg[layer].to(device)
            contrastive_acts.append(pos_act - neg_act)
        
        contrastive_acts_stacked = torch.stack(contrastive_acts)
        
        # Flatten the last two dimensions
        flattened = contrastive_acts_stacked.view(contrastive_acts_stacked.size(0), -1)
        
        steering_vector = steering_vectors[layer].to(device).view(1, -1)  # Flatten steering vector
        
        # Compute cosine similarities with steering vector
        norm_acts = torch.norm(flattened, dim=1, keepdim=True)
        norm_steering = torch.norm(steering_vector)
        cos_sims = torch.mm(flattened, steering_vector.t()) / (norm_acts * norm_steering)
        
        steering_cosine_sims[layer] = cos_sims.squeeze().cpu().numpy()
    
    logging.info("Steering vector cosine similarity computation complete")
    return steering_cosine_sims

def compute_norm_distribution(activations, compute_num_samples):
    logging.info(f"Computing norm distribution using {compute_num_samples} samples")
    norm_dist = {}
    for layer in activations['positive'][0].keys():
        norm_dist[layer] = []
        for pos, neg in zip(activations['positive'][:compute_num_samples], activations['negative'][:compute_num_samples]):
            pos_act = pos[layer].to(device)
            neg_act = neg[layer].to(device)
            contrastive_act = pos_act - neg_act
            norm = torch.norm(contrastive_act).cpu().item()
            norm_dist[layer].append(norm)
    logging.info("Norm distribution computation complete")
    return norm_dist

def plot_distribution(data, ax, title, x_label, y_label, plot_type):
    logging.info(f"Plotting distribution: {title}")
    num_layers = len(data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    all_values = []
    for i, (layer, values) in enumerate(data.items()):
        all_values.extend(values)
        kde = gaussian_kde(values, bw_method=0.05)
        x_range = np.linspace(min(values), max(values), 200)
        ax.plot(x_range, kde(x_range), color=colors[i], label=f'Layer {i+1}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    mean = np.mean(all_values)
    std = np.std(all_values)
    stats_text = f'Mean: {mean:.3f}\nStd: {std:.3f}'
    
    if plot_type == 'cosine':
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:  # norm
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    logging.info(f"Distribution plotting complete: {title}")
    return colors, num_layers

def get_colorbar_ticks(num_layers):
    if num_layers <= 5:
        return list(range(1, num_layers + 1))
    elif num_layers <= 10:
        return [1, num_layers // 2, num_layers]
    else:
        return [1, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers]

def main(load_num_samples, compute_num_samples):
    start_time = time.time()
    logging.info(f"Starting main function with load_num_samples={load_num_samples}, compute_num_samples={compute_num_samples}")
    
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    activations_path = os.path.join(DATA_PATH, "caa_vectors_data", "caa_vectors_activations")
    plots_path = os.path.join(DATA_PATH, "caa_vectors_data", "caa_vectors_plots")
    
    models = ['gemma-2-2b-it', 'gemma-2b-it', 'llama-3-8b-instruct']
    datasets = ["coordinate-other-ais",
                "hallucination",
                "refusal",
                "sycophancy",
                "corrigible-neutral-HHH",
                "myopic-reward",
                "survival-instinct"
            ]
    activation_types = ['penultimate', 'last']

    logging.info(f"Creating figures for {len(models)} models, {len(datasets)} datasets, and {len(activation_types)} activation types")
    fig_cos, axs_cos = plt.subplots(28, 3, figsize=(20, 140))
    fig_norm, axs_norm = plt.subplots(14, 3, figsize=(20, 70))

    for model_idx, model in enumerate(models):
        logging.info(f"Processing model: {model} ({model_idx + 1}/{len(models)})")
        for plot_idx, (dataset, act_type) in enumerate([(d, a) for d in datasets for a in activation_types]):
            logging.info(f"Processing dataset: {dataset}, activation type: {act_type}")
            file_name = f"{model}_{act_type}_activations_{dataset}_{load_num_samples}.pkl"
            file_path = os.path.join(activations_path, dataset, file_name)
            
            activations = load_activations(file_path, load_num_samples)
            steering_vectors = compute_steering_vector(activations, compute_num_samples)
            cosine_sims = compute_cosine_similarity(activations, compute_num_samples)
            steering_cosine_sims = compute_steering_cosine_similarity(activations, steering_vectors, compute_num_samples)
            norm_dist = compute_norm_distribution(activations, compute_num_samples)

            base_title = f"model: {model}, lang={dataset}\ntoken activations: {act_type}, samples={compute_num_samples}"


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
                sm_cos = ScalarMappable(cmap='viridis', norm=Normalize(vmin=1, vmax=num_layers))
                sm_cos.set_array([])
                cbar_cos = plt.colorbar(sm_cos, ax=ax, label='Layer')
                ticks = get_colorbar_ticks(num_layers)
                cbar_cos.set_ticks(ticks)
                cbar_cos.set_ticklabels(ticks)

            logging.info("Plotting norm distribution")
            ax_norm = axs_norm[plot_idx, model_idx]
            colors, num_layers = plot_distribution(norm_dist, ax_norm, base_title, 'Norm', 'Density', 'norm')

            logging.info("Adding colorbar to norm distribution plot")
            sm_norm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=1, vmax=num_layers))
            sm_norm.set_array([])
            cbar_norm = plt.colorbar(sm_norm, ax=ax_norm, label='Layer')
            ticks = get_colorbar_ticks(num_layers)
            cbar_norm.set_ticks(ticks)
            cbar_norm.set_ticklabels(ticks)

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
    fig_cos.savefig(cos_output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Cosine similarity plot saved to: {cos_output_file}")

    norm_output_file = os.path.join(plots_path, f'norm_distribution_contrastive_activations_caa_dataset_{compute_num_samples}.pdf')
    fig_norm.savefig(norm_output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Norm distribution plot saved to: {norm_output_file}")

    plt.close('all')

    end_time = time.time()
    logging.info(f"Script completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot activation distributions with sample size control and GPU acceleration.")
    parser.add_argument("--load_samples", type=int, default=20, help="Number of samples to load from each file")
    parser.add_argument("--compute_samples", type=int, default=20, help="Number of samples to use in computations")
    args = parser.parse_args()

    main(args.load_samples, args.compute_samples)