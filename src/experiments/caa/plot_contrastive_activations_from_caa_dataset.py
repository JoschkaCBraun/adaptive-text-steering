import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the PROJECT_PATH from .env
PROJECT_PATH = os.getenv('PROJECT_PATH')
if not PROJECT_PATH:
    raise ValueError("PROJECT_PATH not found in .env file")

def load_activations(file_path):
    logging.info(f"Loading activations from {file_path}")
    with open(file_path, 'rb') as f:
        activations = pickle.load(f)
    logging.info(f"Loaded activations for {len(activations)} samples")
    return activations

def calculate_contrastive_activations(activations):
    contrastive_activations = []
    for sample in activations:
        contrastive_sample = {}
        for layer in sample['positive'].keys():
            contrastive_sample[layer] = sample['positive'][layer] - sample['negative'][layer]
        contrastive_activations.append(contrastive_sample)
    return contrastive_activations

def calculate_cosine_similarities(contrastive_activations):
    logging.info("Calculating cosine similarities for contrastive activations")
    similarities = {}
    for layer in contrastive_activations[0].keys():
        layer_data = np.array([sample[layer] for sample in contrastive_activations])
        reshaped_data = layer_data.reshape(layer_data.shape[0], -1)
        sim_matrix = cosine_similarity(reshaped_data)
        similarities[layer] = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
    return similarities

def calculate_norms(contrastive_activations):
    logging.info("Calculating norms for contrastive activations")
    norms = {}
    for layer in contrastive_activations[0].keys():
        layer_data = np.array([sample[layer] for sample in contrastive_activations])
        norms[layer] = np.linalg.norm(layer_data.reshape(layer_data.shape[0], -1), axis=1)
    return norms

def plot_distributions(data, datasets, plot_type, output_file, num_samples):
    logging.info(f"Plotting {plot_type} distributions for all datasets")
    fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(6, 32))  # Increased figure height
    fig.suptitle(f"Distribution of Contrastive Activation {plot_type}\nfor All Datasets", fontsize=16, y=0.95)

    for idx, (dataset, values) in enumerate(data.items()):
        ax = axs[idx]
        num_layers = len(values)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / (num_layers - 1)) for i in range(num_layers)]

        if plot_type == 'Cosine Similarity':
            num_comparisons = num_samples * (num_samples - 1) // 2
            ax.set_title(f"{dataset} (num_samples = {num_samples}, num_comparisons = {num_comparisons})")
        else:
            ax.set_title(f"{dataset} (num_samples = {num_samples})")

        ax.set_xlabel(f"Contrastive Activation {plot_type}")
        ax.set_ylabel("Density")

        for layer_idx, (layer, layer_values) in enumerate(values.items()):
            # Adjust bandwidth for smoothness
            if plot_type == 'Norm':
                kde = gaussian_kde(layer_values, bw_method=0.1)  # Increased smoothness
            else:  # Cosine Similarity
                kde = gaussian_kde(layer_values, bw_method=0.05)  # Decreased smoothness

            x_range = np.linspace(min(layer_values), max(layer_values), 200)
            ax.plot(x_range, kde(x_range), color=colors[layer_idx], label=f"Layer {int(layer)+1}")

        if plot_type == 'Cosine Similarity':
            ax.set_xlim(-1, 1)

        # Add color bar for each subplot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_layers))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, aspect=30)
        cbar.set_label('Layer')
        cbar.set_ticks(np.arange(1, num_layers + 1))  # 18 ticks for layers
        cbar.set_ticklabels([f'{int(x)}' for x in range(1, num_layers + 1)])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved to {output_file}")

def main():
    input_folder = os.path.join(PROJECT_PATH, "data", "caa_vectors_data", "caa_vectors_activations")
    output_folder = os.path.join(PROJECT_PATH, "data", "caa_vectors_data", "caa_vectors_plots_claude")
    os.makedirs(output_folder, exist_ok=True)

    datasets = [
        "coordinate-other-ais",
        "hallucination",
        "refusal",
        "sycophancy",
        "corrigible-neutral-HHH",
        "myopic-reward",
        "survival-instinct"
    ]

    num_samples = 250

    cosine_similarities_data = {}
    norms_data = {}

    for dataset in datasets:
        input_file = os.path.join(input_folder, f"caa_{dataset}_activations_en_{num_samples}_for_gemma_2b_last_2.pkl")
        activations = load_activations(input_file)
        contrastive_activations = calculate_contrastive_activations(activations)

        cosine_similarities_data[dataset] = calculate_cosine_similarities(contrastive_activations)
        norms_data[dataset] = calculate_norms(contrastive_activations)

    plot_distributions(cosine_similarities_data, datasets, 'Cosine Similarity', 
                       os.path.join(output_folder, f"all_datasets_contrastive_activation_cosine_similarities_distribution.pdf"),
                       num_samples)
    
    plot_distributions(norms_data, datasets, 'Norm', 
                       os.path.join(output_folder, f"all_datasets_contrastive_activation_norms_distribution.pdf"),
                       num_samples)

    logging.info(f"All plots saved to {output_folder}")

if __name__ == "__main__":
    main()