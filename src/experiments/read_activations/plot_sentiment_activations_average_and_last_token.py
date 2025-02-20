import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_activations(file_path):
    logging.info(f"Loading activations from {file_path}")
    with open(file_path, 'rb') as f:
        activations = pickle.load(f)
    logging.info(f"Loaded activations for {len(activations)} samples")
    return activations

def calculate_cosine_similarities(activations):
    similarities = {}
    for layer in activations[0].keys():
        layer_data = torch.stack([sample[layer] for sample in activations]).cpu().numpy()
        reshaped_data = layer_data.reshape(layer_data.shape[0], -1)
        sim_matrix = cosine_similarity(reshaped_data)
        similarities[layer] = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
    return similarities

def calculate_pos_neg_cosine_similarities(positive_activations, negative_activations):
    similarities = {}
    for layer in positive_activations[0].keys():
        pos_data = torch.stack([sample[layer] for sample in positive_activations]).cpu().numpy()
        neg_data = torch.stack([sample[layer] for sample in negative_activations]).cpu().numpy()
        pos_reshaped = pos_data.reshape(pos_data.shape[0], -1)
        neg_reshaped = neg_data.reshape(neg_data.shape[0], -1)
        similarities[layer] = np.array([cosine_similarity(p.reshape(1, -1), n.reshape(1, -1))[0][0] 
                                        for p, n in zip(pos_reshaped, neg_reshaped)])
    return similarities

def calculate_norms(activations):
    norms = {}
    for layer in activations[0].keys():
        layer_data = torch.stack([sample[layer] for sample in activations]).cpu().numpy()
        norms[layer] = np.linalg.norm(layer_data.reshape(layer_data.shape[0], -1), axis=1)
    return norms
def add_colorbar_legend(fig, num_layers):
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=1, vmax=num_layers)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Layer')
    cb.set_ticks(range(1, num_layers + 1))
    cb.set_ticklabels(range(1, num_layers + 1))

def plot_cosine_similarities(positive_similarities, negative_similarities, pos_neg_similarities, datasets, output_file, num_samples):
    num_comparisons = num_samples * (num_samples - 1) // 2
    fig, axs = plt.subplots(len(datasets), 1, figsize=(12, 6*len(datasets)), squeeze=False)
    fig.suptitle(f"Cosine Similarities of Activations\n(Samples: {num_samples}, Comparisons: {num_comparisons})", fontsize=16)

    for idx, dataset in enumerate(datasets):
        ax = axs[idx, 0]
        pos_sims = positive_similarities[dataset]
        neg_sims = negative_similarities[dataset]
        pos_neg_sims = pos_neg_similarities[dataset]

        num_layers = len(pos_sims)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / (num_layers - 1)) for i in range(num_layers)]

        for layer_idx, (layer, pos_values) in enumerate(pos_sims.items()):
            neg_values = neg_sims[layer]
            pos_neg_values = pos_neg_sims[layer]
            kde_pos = gaussian_kde(pos_values)
            kde_neg = gaussian_kde(neg_values)
            kde_pos_neg = gaussian_kde(pos_neg_values)
            x_range = np.linspace(-1, 1, 200)
            ax.plot(x_range, kde_pos(x_range), color=colors[layer_idx], label=f"Positive (Layer {int(layer)+1})")
            ax.plot(x_range, kde_neg(x_range), color=colors[layer_idx], linestyle='--', label=f"Negative (Layer {int(layer)+1})")
            ax.plot(x_range, kde_pos_neg(x_range), color=colors[layer_idx], linestyle=':', label=f"Pos-Neg (Layer {int(layer)+1})")

        ax.set_title(f"{dataset}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_xlim(-1, 1)

    add_colorbar_legend(fig, num_layers)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Cosine similarities plot saved to {output_file}")

def plot_norm_distributions(positive_norms, negative_norms, datasets, output_file, num_samples):
    fig, axs = plt.subplots(len(datasets), 1, figsize=(12, 6*len(datasets)), squeeze=False)
    fig.suptitle(f"Norm Distributions of Positive and Negative Activations\n(Samples: {num_samples})", fontsize=16)

    for idx, dataset in enumerate(datasets):
        ax = axs[idx, 0]
        pos_norms = positive_norms[dataset]
        neg_norms = negative_norms[dataset]

        num_layers = len(pos_norms)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / (num_layers - 1)) for i in range(num_layers)]

        for layer_idx, (layer, pos_values) in enumerate(pos_norms.items()):
            neg_values = neg_norms[layer]
            kde_pos = gaussian_kde(pos_values)
            kde_neg = gaussian_kde(neg_values)
            x_range = np.linspace(min(pos_values.min(), neg_values.min()),
                                  max(pos_values.max(), neg_values.max()), 200)
            ax.plot(x_range, kde_pos(x_range), color=colors[layer_idx])
            ax.plot(x_range, kde_neg(x_range), color=colors[layer_idx], linestyle='--')

        ax.set_title(f"{dataset}")
        ax.set_xlabel("Norm")
        ax.set_ylabel("Density")

    add_colorbar_legend(fig, num_layers)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Norm distributions plot saved to {output_file}")

def main():
    # Get the PROJECT_PATH from .env
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    read_sentiment_activations_folder = os.path.join(PROJECT_PATH, "data", "read_activations", "read_sentiment_activations")

    input_folder = os.path.join(read_sentiment_activations_folder, "sentiment_activations")
    output_folder = os.path.join(read_sentiment_activations_folder, "sentiment_plots")
    os.makedirs(output_folder, exist_ok=True)

    datasets = ["en", "de", "fr", "random"]
    num_samples = 100
    model_name = "gemma2_2b_it"

    positive_similarities = {}
    negative_similarities = {}
    pos_neg_similarities = {}
    positive_norms = {}
    negative_norms = {}

    for dataset in datasets:
        input_file = os.path.join(input_folder, f"{model_name}_all_activations_{dataset}_{num_samples}.pkl")
        logging.info(f"Loading activations from {input_file}")
        
        try:
            activations = load_activations(input_file)
        except FileNotFoundError:
            logging.error(f"File not found: {input_file}")
            continue

        positive_activations = activations['positive']
        negative_activations = activations['negative']

        positive_similarities[dataset] = calculate_cosine_similarities(positive_activations)
        negative_similarities[dataset] = calculate_cosine_similarities(negative_activations)
        pos_neg_similarities[dataset] = calculate_pos_neg_cosine_similarities(positive_activations, negative_activations)

        positive_norms[dataset] = calculate_norms(positive_activations)
        negative_norms[dataset] = calculate_norms(negative_activations)

    # Plot all data after processing all datasets
    plot_cosine_similarities(positive_similarities, negative_similarities, pos_neg_similarities, datasets, 
                             os.path.join(output_folder, f"{model_name}_cosine_similarities_{num_samples}_samples.pdf"),
                             num_samples)

    plot_norm_distributions(positive_norms, negative_norms, datasets,
                            os.path.join(output_folder, f"{model_name}_norm_distributions_{num_samples}_samples.pdf"),
                            num_samples)

    logging.info("All plots generated successfully.")

if __name__ == "__main__":
    main()