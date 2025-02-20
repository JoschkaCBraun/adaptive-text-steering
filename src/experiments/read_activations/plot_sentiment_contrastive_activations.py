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

def plot_contrastive_activation_norms_and_similarities(contrastive_activations, output_folder, num_samples, num_layers=18):
    layers = sorted(contrastive_activations.keys(), key=lambda x: int(x.split('_')[1]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Contrastive Activation Analysis (Samples: {num_samples})", fontsize=16)
    
    # Create color map
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (num_layers - 1)) for i in range(num_layers)]
    
    # Plot 1: Distribution of Contrastive Activation Norms
    ax1.set_title("Distribution of Contrastive Activation Norms Across Layers")
    ax1.set_xlabel("Norm of Contrastive Activation")
    ax1.set_ylabel("Density")
    
    for idx, layer in enumerate(layers):
        norms = [np.linalg.norm(act) for act in contrastive_activations[layer]]
        kde = gaussian_kde(norms)
        x_range = np.linspace(min(norms), max(norms), 100)
        ax1.plot(x_range, kde(x_range), color=colors[idx])
    
    # Plot 2: Alignment with Mean Contrastive Activation
    ax2.set_title("Alignment with Mean Contrastive Activation Across Layers")
    ax2.set_xlabel("Cosine Similarity to Mean Contrastive Activation")
    ax2.set_ylabel("Density")
    
    mean_activations = {layer: np.mean(activations, axis=0) for layer, activations in contrastive_activations.items()}
    
    for idx, layer in enumerate(layers):
        activations = np.array(contrastive_activations[layer])
        mean_activation = mean_activations[layer]
        
        similarities = cosine_similarity(activations, [mean_activation]).flatten()
        kde = gaussian_kde(similarities)
        x_range = np.linspace(-1, 1, 200)
        density = kde(x_range)
        
        ax2.plot(x_range, density, color=colors[idx])
    
    ax2.set_xlim(-1, 1)
    
    # Add colorbar legend
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_layers))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks(np.linspace(1, num_layers, num_layers))
    cbar.set_ticklabels(range(1, num_layers + 1))
    cbar.set_label('Layer')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = os.path.join(output_folder, f'contrastive_activation_analysis_{num_layers}_{num_samples}.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main():
    input_file = "data/read_activations/read_sentiment_activations/gemma_2b_all_layers_500_sentiment_activations.pkl"
    output_folder = "data/read_activations/read_sentiment_activations/sentiment_activations_plots"
    os.makedirs(output_folder, exist_ok=True)
    
    logging.info(f"Loading activations from {input_file}")
    activations = load_activations(input_file)
    
    logging.info(f"Activation data structure: {type(activations)}")
    logging.info(f"Keys in activation data: {activations.keys()}")
    
    num_layers = len(activations)
    logging.info(f"Number of layers detected: {num_layers}")
    
    # Assuming the first layer's data is representative of all layers
    first_layer_key = list(activations.keys())[0]
    num_samples = len(activations[first_layer_key])
    logging.info(f"Number of samples detected: {num_samples}")
    
    logging.info(f"Plotting contrastive activation norms and similarities")
    plot_contrastive_activation_norms_and_similarities(activations, output_folder, num_samples, num_layers)
    
    logging.info("Plotting completed")

if __name__ == "__main__":
    main()