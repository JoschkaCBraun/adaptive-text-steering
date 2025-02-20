"""
Plot cosine similarities between paired activations and their steering vectors,
using data loaded from HDF5 files via the new file structure.

This script produces one plot per pairing (prompt) type. In each plot, the cosine
similarity distributions for all datasets are shown, with colors mapped from green
(low dataset index) to red (high dataset index).
"""

import os
import logging
import matplotlib.pyplot as plt
import torch

import src.utils as utils
from src.utils.dataset_names import all_datasets_figure_13

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_pairing(dataset_name: str, pairing_type: str, device: torch.device) -> list:
    """
    Load paired activations for a given dataset and pairing type, compute the contrastive
    activations, compute the steering vector, and calculate cosine similarities between
    each contrastive activation and the steering vector.
    
    Args:
        dataset_name: Name of the dataset to load.
        pairing_type: The pairing type to use (e.g., "prefilled_answer", "instruction", etc.).
        device: Torch device to use for computation.
    
    Returns:
        List of cosine similarity values.
    """
    # Load 200 samples using the new loader function
    data = utils.load_paired_activations_and_logits(
        dataset_name=dataset_name,
        pairing_type=pairing_type,
        num_samples=200
    )

    # Extract activations as NumPy arrays
    matching_acts_np = data['matching_activations']       # shape: [num_samples, 4096]
    non_matching_acts_np = data['non_matching_activations']  # shape: [num_samples, 4096]

    # Convert each row to a PyTorch tensor (list of 1D tensors)
    matching_acts = [torch.tensor(row, dtype=torch.float32) for row in matching_acts_np]
    non_matching_acts = [torch.tensor(row, dtype=torch.float32) for row in non_matching_acts_np]

    # Compute contrastive activations using the provided utility function
    contrastive_activations = utils.compute_contrastive_activations(
        positive_activations=matching_acts,
        negative_activations=non_matching_acts,
        device=device
    )

    # Compute the steering vector from the contrastive activations
    steering_vector = utils.compute_contrastive_steering_vector(
        positive_activations=matching_acts,
        negative_activations=non_matching_acts,
        device=device
    )

    # Compute cosine similarities between each contrastive activation and the steering vector
    steering_sims = utils.compute_many_against_one_cosine_similarity(
        activations=contrastive_activations,
        steering_vector=steering_vector,
        device=device
    )

    return steering_sims


def plot_distributions_for_pairing(pairing_type: str, device: torch.device, output_dir: str) -> None:
    """
    For a given pairing type, iterate over all datasets and plot the distribution
    of cosine similarities between the contrastive activations and their steering vector.
    
    Each dataset's distribution is color-coded using a continuous colormap from green
    (low dataset index) to red (high dataset index).
    
    Args:
        pairing_type: The pairing type (prompt type) to process.
        device: Torch device for computation.
        output_dir: Directory where the plot PDF will be saved.
    """
    datasets = all_datasets_figure_13
    num_datasets = len(datasets)
    
    # Set up the colormap: using RdYlGn_r so that 0.0 is green and 1.0 is red.
    cmap = plt.get_cmap("RdYlGn_r")
    
    # Create a new figure for this pairing type.
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Steering Vector Cosine Similarities\nPairing Type: {pairing_type}")
    ax.set_xlim(-1, 1)
    
    for idx, dataset_name in enumerate(datasets):
        try:
            steering_sims = process_pairing(dataset_name, pairing_type, device=device)
        except Exception as e:
            logging.error(f"Error processing dataset '{dataset_name}' for pairing '{pairing_type}': {str(e)}")
            continue
        
        # Determine color based on dataset index
        color = cmap(idx / (num_datasets - 1))  # low index: green; high index: red
        
        # Plot histogram for this dataset
        ax.hist(steering_sims, bins=20, density=True, alpha=0.6,
                color=color, histtype='step', linewidth=2)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"cosine_similarity_{pairing_type}_200_samples.pdf")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot for pairing type '{pairing_type}' to {output_path}")


def main() -> None:
    # Get the computation device from utils.
    device = utils.get_device()

    # Get DATA_PATH and construct the plots directory.
    DATA_PATH = utils.get_path('DATA_PATH')
    PLOTS_PATH = os.path.join(DATA_PATH, "0_paper_plots")
    os.makedirs(PLOTS_PATH, exist_ok=True)

    # List of pairing types to iterate over (as defined in the JSON pairings)
    pairing_types = [
        "prefilled_answer",
        "instruction",
        "5-shot",
        "prefilled_instruction",
        "prefilled_5-shot",
        "instruction_5-shot",
        "prefilled_instruction_5-shot"
    ]

    # For each pairing type, create a separate plot combining all datasets.
    for pairing_type in pairing_types:
        logging.info(f"Processing pairing type: {pairing_type}")
        plot_distributions_for_pairing(pairing_type, device, PLOTS_PATH)


if __name__ == "__main__":
    main()
