"""
plot_cosine_similarity_distribution.py
"""

import os
import pickle
import json
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import src.utils.dataset_names as dn
import src.utils as utils
import matplotlib.patches as mpatches  # NEW: for creating legend patches

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = utils.get_device()

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

def plot_distribution_all(results: dict, ax: plt.Axes) -> None:
    """Overlay cosine similarity distributions for all datasets in one plot,
    plot the maximum density at each x-axis position as a line, and add a
    secondary legend that groups the 36 datasets into 6 groups with the mean
    cosine similarity.
    
    results: dict mapping dataset name to list of cosine similarity values.
    """
    # Get the dataset names in their natural order
    dataset_names = list(results.keys())
    num_datasets = len(dataset_names)
    
    # Define common bins for the histograms (we use 200 bins between -1 and 1)
    bins = np.linspace(-1, 1, 201)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Use a colormap that goes from green (low index) to red (high index)
    cmap = plt.cm.RdYlGn_r
    colors = cmap(np.linspace(0, 1, num_datasets))
    
    # Compute density arrays for each dataset for later use in the max density line.
    density_list = []
    for dataset in dataset_names:
        values = results[dataset]
        density, _ = np.histogram(values, bins=bins, density=True)
        density_list.append(density)
    density_array = np.vstack(density_list)
    max_density = np.max(density_array, axis=0)
    
    # Plot the histograms in reversed order so that lower dataset indices appear on top.
    for i in reversed(range(num_datasets)):
        dataset = dataset_names[i]
        values = results[dataset]
        # Draw filled histogram with colored edges. Each histogram is labeled.
        ax.hist(values, bins=bins, density=True, alpha=0.5,
                color=colors[i], edgecolor=colors[i],
                label=f"Dataset {i+1}", histtype='bar', linewidth=1.5)
    
    # Update x-axis label and other plot settings.
    ax.set_xlabel('Cosine similarity between activation differences and steering vector')
    ax.set_xlim(-1, 1)
    ax.set_ylabel('Density')
    ax.set_title('Directional agreement of activation differences across datasets')
    
    # Create a colorbar with fewer ticks (5 evenly spaced ticks).
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=num_datasets - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Dataset steerability rank')
    ticks = np.unique(np.round(np.linspace(0, num_datasets - 1, 5)).astype(int))
    cbar.set_ticks(ticks)
    # Shift the tick labels up by 1
    cbar.set_ticklabels((ticks + 1).tolist())

    # NEW: Create a new legend that groups the 36 datasets into 6 groups of 6.
    group_handles = []
    num_groups = 6  # since there are 36 datasets
    for group_index in range(num_groups):
        group_start_index = group_index * 6
        group_end_index = group_start_index + 6
        # Select the dataset names for this group.
        group_dataset_names = dataset_names[group_start_index:group_end_index]
        # Concatenate all cosine similarity values from these 6 datasets.
        group_values = []
        for ds in group_dataset_names:
            group_values.extend(results[ds])
        group_mean = np.mean(group_values)
        # Choose the median color of this group (for example, the color at index group_start_index + 2).
        median_color = colors[group_start_index + 2]
        # Create a patch with the same alpha and edge color.
        patch = mpatches.Patch(color=median_color, alpha=1.0,
                                label=f"{group_mean:.2f} (group {group_start_index+1}-{group_end_index})",
                                ec=median_color, linewidth=1.5)
        group_handles.append(patch)
    
    # Add the new legend in the upper right.
    group_legend = ax.legend(handles=group_handles, title="Mean cosine similarity", loc='upper left', fontsize=8, title_fontsize=9)
    ax.add_artist(group_legend)

def main() -> None:
    compute_num_samples = 500
    load_num_samples = 500
    DATA_PATH = os.getenv('DATA_PATH')
    DISK_PATH = os.getenv('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(DATA_PATH, "0_paper_plots")
    
    # Create intermediate results directory
    intermediate_dir = os.path.join(plots_path, "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)
    # Intermediate JSON filename
    intermediate_json_file = os.path.join(
        intermediate_dir, f"activation_cosine_similarity_llama2_7b_chat_{load_num_samples}_samples.json"
    )
    
    selected_layers = [13]
    model = ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', selected_layers)
    # Use all datasets without limit
    datasets = dn.all_datasets_figure_13
    activation_type = "last"  # Only consider "last"
    
    # Check if intermediate results JSON exists. If yes, load; if no, compute.
    if os.path.exists(intermediate_json_file):
        logging.info(f"Loading intermediate results from {intermediate_json_file}")
        with open(intermediate_json_file, "r") as f:
            cosine_sim_results = json.load(f)
    else:
        logging.info("Intermediate results not found. Running cosine similarity calculations...")
        cosine_sim_results = {}
        # Process each dataset
        for dataset in datasets:
            file_name = f"{model[0]}_activations_{dataset}_for_{load_num_samples}_samples_{activation_type}.pkl"
            file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
            logging.info(f"Processing dataset: {dataset}")
            activations = load_activations(file_path, load_num_samples, selected_layers)
            contrastive_activations = utils.compute_contrastive_activations_dict(
                activations['positive'], activations['negative'], device)
            steering_vectors = utils.compute_contrastive_steering_vector_dict(
                activations['positive'], activations['negative'], device)
            # Compute cosine similarities between contrastive activations and steering vectors.
            steering_cosine_sims = utils.compute_many_against_one_cosine_similarity_dict(
                contrastive_activations, steering_vectors, device)
            # Since only layer 13 is used, extract the cosine similarity list for that layer.
            cosine_sim_results[dataset] = steering_cosine_sims.get(13, [])
        
        # Save the results to JSON.
        logging.info(f"Saving intermediate results to {intermediate_json_file}")
        with open(intermediate_json_file, "w") as f:
            json.dump(cosine_sim_results, f)
    
    # Create a single combined plot with the specified figure size.
    fig, ax = plt.subplots(figsize=(7, 2.5))
    plot_distribution_all(cosine_sim_results, ax)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(plots_path, exist_ok=True)
    output_file = os.path.join(plots_path, f'activation_cosine_similarity_{compute_num_samples}.pdf')
    fig.savefig(output_file, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
