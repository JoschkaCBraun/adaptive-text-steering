import os
import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_activations(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_steering_vectors(positive_activations, negative_activations):
    Delta = positive_activations - negative_activations
    s_method1 = np.median(Delta, axis=0)  # Median of Differences
    s_method2 = np.median(positive_activations, axis=0) - np.median(negative_activations, axis=0)  # Difference of Medians
    s_mean_unnormalized = np.mean(Delta, axis=0)  # Mean Unnormalized
    norms = np.linalg.norm(Delta, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_differences = Delta / norms[:, np.newaxis]
    s_mean_normalized = np.mean(normalized_differences, axis=0)  # Mean Normalized
    return s_method1, s_method2, s_mean_unnormalized, s_mean_normalized

def plot_similarities(activations_dict, output_folder, num_samples, num_layers=18):
    datasets = {'en': 'English', 'de': 'German', 'fr': 'French', 'random': 'Random'}
    steering_methods = ['median_of_differences', 'difference_of_medians', 'mean_unnormalized', 'mean_normalized']
    method_titles = {
        'median_of_differences': 'Median of Differences',
        'difference_of_medians': 'Difference of Medians',
        'mean_unnormalized': 'Mean Unnormalized',
        'mean_normalized': 'Mean Normalized'
    }

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(f"Cosine Similarity for All Steering Vector Methods (Samples: {num_samples})", fontsize=16)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (num_layers - 1)) for i in range(num_layers)]

    for dataset_idx, (short_name, full_name) in enumerate(datasets.items()):
        activations = activations_dict[short_name]
        layers = sorted(activations.keys(), key=lambda x: int(x.split('_')[1]))

        # Random dataset is used as negative in comparisons
        negative_activations = activations_dict['random']

        for layer_idx, layer in enumerate(layers):
            positive_activations = np.array(activations[layer])
            negative_activations_layer = np.array(negative_activations[layer])
            noise = np.random.normal(0, 1e-6, negative_activations_layer.shape)
            negative_activations_layer = np.array(negative_activations[layer]) + noise

            # Compute all steering vectors
            s_method1, s_method2, s_mean_unnormalized, s_mean_normalized = compute_steering_vectors(positive_activations, negative_activations_layer)
            steering_vectors = [s_method1, s_method2, s_mean_unnormalized, s_mean_normalized]

            for method_idx, (method, steering_vector) in enumerate(zip(steering_methods, steering_vectors)):
                ax = axs[dataset_idx, method_idx]
                sims = cosine_similarity(positive_activations, [steering_vector]).flatten()

                # Check if variance is too small, skip KDE if true
                if np.var(sims) < 1e-6:
                    logging.warning(f"Variance too small for layer {layer} in {full_name} using method {method}. Skipping KDE.")
                    continue

                kde = gaussian_kde(sims)
                x_range = np.linspace(-1, 1, 200)
                try:
                    ax.plot(x_range, kde(x_range), color=colors[layer_idx], label=f'Layer {layer_idx + 1}')
                except np.linalg.LinAlgError:
                    logging.warning(f"Singular covariance matrix for {method} in layer {layer}. Skipping this plot.")
                    continue

                if layer_idx == len(layers) - 1:
                    ax.set_title(f"{full_name} - {method_titles[method]}")
                    ax.set_xlim(-1, 1)
                    ax.set_xlabel('Cosine Similarity')
                    ax.set_ylabel('Density')
                    add_colorbar(ax, cmap, num_layers)

    # Save plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = os.path.join(output_folder, f'cosine_similarity_steering_vectors_{num_samples}_samples_comparison.pdf')
    fig.savefig(output_file, bbox_inches='tight')

    logging.info(f"Plot saved to {output_file}")

def add_colorbar(ax, cmap, num_layers):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_layers))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks(np.linspace(1, num_layers, num_layers))
    cbar.set_ticklabels(range(1, num_layers + 1))
    cbar.set_label('Layer')

def plot_vector_alignment(steering_vectors_dict, output_folder, num_layers):
    datasets = ['en', 'de', 'fr', 'random']
    methods = ['median_of_differences', 'difference_of_medians', 'mean_unnormalized', 'mean_normalized']
    dataset_titles = ['English', 'German', 'French', 'Random']
    method_titles = ['Median of Differences', 'Difference of Medians', 'Mean Unnormalized', 'Mean Normalized']

    colors = ['b', 'g', 'r', 'c']  # Colors for datasets
    linestyles = ['-', '--', ':', '-.']  # Line styles for methods

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    fig.suptitle("Cosine Similarity of Steering Vectors Across Layers", fontsize=16)

    axes = [ax1, ax2]

    for dataset_idx, dataset in enumerate(datasets):
        for method_idx, method in enumerate(methods):
            similarities = []
            for layer_idx in range(num_layers - 1):  # Compare layer i to layer i+1
                vec_current = steering_vectors_dict[dataset][method][layer_idx]
                vec_next = steering_vectors_dict[dataset][method][layer_idx + 1]
                sim = cosine_similarity([vec_current], [vec_next]).flatten()[0]
                similarities.append(sim)
            
            for ax in axes:
                ax.plot(range(num_layers - 1), similarities, 
                        color=colors[dataset_idx], 
                        linestyle=linestyles[method_idx], 
                        label=f'{dataset_titles[dataset_idx]} - {method_titles[method_idx]}')

    # Set y-axis limits and ticks
    ax1.set_ylim(0.78, 1.0)
    ax2.set_ylim(-0.03, 0.03)

    # Hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Add broken axis marks
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Labels and ticks
    ax2.set_xlabel("Layer", fontsize=12)
    # fig.text(0.04, 0.5, "Cosine Similarity", va='center', rotation='vertical', fontsize=12)
    ax1.set_xticks([])  # Remove x-ticks from top subplot
    ax2.set_xticks(range(num_layers - 1))

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)

    # Grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_folder, 'vector_alignment_across_layers_broken_axis.pdf')
    fig.savefig(output_file, bbox_inches='tight')
    logging.info(f"Layer alignment plot with broken axis saved to {output_file}")
    plt.close(fig)

def plot_cosine_similarity_matrix(steering_vectors_dict, output_folder, num_layers):
    datasets = ['en', 'de', 'fr', 'random']
    methods = ['median_of_differences', 'difference_of_medians', 'mean_unnormalized', 'mean_normalized']
    
    # Create a figure with a gridspec for the plots
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(3, 6)  # 3 rows, 6 columns
    fig.suptitle("Cosine Similarity Matrix Across Layers", fontsize=16)

    # Create axes for the plots
    axs = [[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(3)]

    # Plot the cosine similarity matrices
    for layer_idx in range(num_layers):
        row = layer_idx // 6
        col = layer_idx % 6

        # Initialize a 16x16 matrix for each layer
        similarity_matrix = np.zeros((16, 16))

        # Compute cosine similarity for all vector pairs
        for i, (dataset1, method1) in enumerate([(d, m) for d in datasets for m in methods]):
            vec1 = steering_vectors_dict[dataset1][method1][layer_idx]
            for j, (dataset2, method2) in enumerate([(d, m) for d in datasets for m in methods]):
                vec2 = steering_vectors_dict[dataset2][method2][layer_idx]
                similarity_matrix[i, j] = cosine_similarity([vec1], [vec2]).flatten()[0]

        # Plot the similarity matrix in the corresponding subplot
        ax = axs[row][col]
        im = ax.matshow(similarity_matrix, cmap='viridis')
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=10)

        # Remove x and y axis labels
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout and add a colorbar
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # Create a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)

    # Save the combined plot
    output_file = os.path.join(output_folder, 'combined_cosine_similarity_matrix.pdf')
    fig.savefig(output_file, bbox_inches='tight')
    logging.info(f"Combined cosine similarity matrix plot saved to {output_file}")

    plt.close(fig)



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

    methods = ['median_of_differences', 'difference_of_medians', 'mean_unnormalized', 'mean_normalized']
    layers = sorted(activations_dict['en'].keys(), key=lambda x: int(x.split('_')[1]))

    steering_vectors_dict = {
        dataset: {method: [] for method in methods} for dataset in datasets
    }

    for dataset in datasets:
        activations = activations_dict[dataset]
        for layer in layers:
            positive_activations = np.array(activations[layer])
            negative_activations_layer = np.array(activations_dict['random'][layer])

            s_method1, s_method2, s_mean_unnormalized, s_mean_normalized = compute_steering_vectors(
                positive_activations, negative_activations_layer
            )
            steering_vectors_dict[dataset]['median_of_differences'].append(s_method1)
            steering_vectors_dict[dataset]['difference_of_medians'].append(s_method2)
            steering_vectors_dict[dataset]['mean_unnormalized'].append(s_mean_unnormalized)
            steering_vectors_dict[dataset]['mean_normalized'].append(s_mean_normalized)

    # Plot vector alignment across layers
    plot_vector_alignment(steering_vectors_dict, output_folder, len(layers))

    # Plot cosine similarity matrix for each layer
    plot_cosine_similarity_matrix(steering_vectors_dict, output_folder, len(layers))

if __name__ == "__main__":
    main()
