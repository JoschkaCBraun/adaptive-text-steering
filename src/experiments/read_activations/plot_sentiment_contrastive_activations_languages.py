import os
import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_activations(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_norms_and_similarities(activations_dict, output_folder, num_samples, num_layers=18):
    datasets = {'en': 'English', 'de': 'German', 'fr': 'French', 'random': 'Random'}
    
    # Create subplots for norms and similarities
    fig_norms, axs_norms = plt.subplots(2, 2, figsize=(12, 10))
    fig_sims, axs_sims = plt.subplots(2, 2, figsize=(12, 10))
    fig_norms.suptitle(f"Contrastive Activation Norms (Samples: {num_samples})", fontsize=16)
    fig_sims.suptitle(f"Cosine Similarity to Mean Activation (Samples: {num_samples})", fontsize=16)
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (num_layers - 1)) for i in range(num_layers)]

    for idx, (short_name, full_name) in enumerate(datasets.items()):
        activations = activations_dict[short_name]
        layers = sorted(activations.keys(), key=lambda x: int(x.split('_')[1]))

        # Select subplot for norms and similarities
        ax_norms = axs_norms[idx // 2, idx % 2]
        ax_sims = axs_sims[idx // 2, idx % 2]
        ax_norms.set_title(f"Distribution of L2 Norms of {full_name} Contrastive Activations")
        ax_sims.set_title(f"Cosine Similarity of {full_name} Contrastive Activations")
        
        # Norms
        for layer_idx, layer in enumerate(layers):
            norms = [np.linalg.norm(act) for act in activations[layer]]
            kde = gaussian_kde(norms)
            x_range = np.linspace(min(norms), max(norms), 100)
            ax_norms.plot(x_range, kde(x_range), color=colors[layer_idx], label=f'Layer {layer_idx + 1}')
        
        # Add legend for layers
        divider = make_axes_locatable(ax_norms)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_layers))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_ticks(np.linspace(1, num_layers, num_layers))
        cbar.set_ticklabels(range(1, num_layers + 1))
        cbar.set_label('Layer')
                
        # Similarities
        mean_activations = {layer: np.mean(acts, axis=0) for layer, acts in activations.items()}
        for layer_idx, layer in enumerate(layers):
            acts = np.array(activations[layer])
            mean_activation = mean_activations[layer]
            sims = cosine_similarity(acts, [mean_activation]).flatten()
            kde = gaussian_kde(sims)
            x_range = np.linspace(-1, 1, 200)
            ax_sims.plot(x_range, kde(x_range), color=colors[layer_idx], label=f'Layer {layer_idx + 1}')

        ax_sims.set_xlim(-1, 1)

        # Add legend for layers
        divider = make_axes_locatable(ax_sims)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_layers))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_ticks(np.linspace(1, num_layers, num_layers))
        cbar.set_ticklabels(range(1, num_layers + 1))
        cbar.set_label('Layer')
    
    # Save plots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    norms_output_file = os.path.join(output_folder, f'contrastive_activation_norms_{num_samples}_samples_language_comparison.pdf')
    sims_output_file = os.path.join(output_folder, f'contrastive_activation_similarities_{num_samples}_samples_language_comparison.pdf')
    fig_norms.savefig(norms_output_file, bbox_inches='tight')
    fig_sims.savefig(sims_output_file, bbox_inches='tight')
    
    logging.info(f"Norms plot saved to {norms_output_file}")
    logging.info(f"Similarities plot saved to {sims_output_file}")

def main():
    input_folder = "data/read_activations/read_sentiment_activations/sentiment_activations/"
    output_folder = "data/read_activations/read_sentiment_activations/sentiment_activations_plots/"
    os.makedirs(output_folder, exist_ok=True)
    
    datasets = ['en', 'de', 'fr', 'random']
    activations_dict = {}
    
    # Load activations for all datasets
    for dataset in datasets:
        input_file = os.path.join(input_folder, f"gemma_2b_all_layers_500_sentiment_activations_{dataset}.pkl")
        logging.info(f"Loading activations from {input_file}")
        activations_dict[dataset] = load_activations(input_file)
    
    # Calculate number of layers and samples based on the first dataset
    first_dataset = datasets[0]
    num_layers = len(activations_dict[first_dataset])
    num_samples = len(activations_dict[first_dataset][list(activations_dict[first_dataset].keys())[0]])
    
    logging.info(f"Plotting norms and similarities for {num_samples} samples across {num_layers} layers")
    plot_norms_and_similarities(activations_dict, output_folder, num_samples, num_layers)
    
    logging.info("Plotting completed")

if __name__ == "__main__":
    main()
