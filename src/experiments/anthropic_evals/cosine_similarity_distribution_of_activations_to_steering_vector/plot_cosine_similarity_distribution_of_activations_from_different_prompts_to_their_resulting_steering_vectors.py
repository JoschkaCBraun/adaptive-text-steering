"""
Plot cosine similarities between paired activations and their steering vectors,
using data loaded from HDF5 files via the new file structure.
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

import src.utils as utils
from src.utils.dataset_names import all_datasets_figure_13

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_pairing(
    dataset_name: str,
    pairing_type: str,
    device: torch.device
) -> list:
    """
    Load paired activations using the provided pairing type, compute the contrastive
    activations, compute the steering vector, and then calculate cosine similarities
    between each contrastive activation and the steering vector.

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
    matching_acts_np = data['matching_activations']      # shape: [num_samples, 4096]
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


def plot_distributions_for_dataset(
    dataset_name: str,
    ax: plt.Axes,
    device: torch.device
) -> None:
    """
    For a given dataset, iterate over all pairing types and plot the distribution
    of cosine similarities between the contrastive activations and their steering vector.
    
    Args:
        dataset_name: Name of the dataset.
        ax: Matplotlib Axes on which to plot.
        device: Torch device for computation.
    """
    # List of pairing types to iterate over (as specified in the JSON)
    pairing_types = [
        "prefilled_answer",
        "instruction",
        "5-shot",
        "prefilled_instruction",
        "prefilled_5-shot",
        "instruction_5-shot",
        "prefilled_instruction_5-shot"
    ]

    # Use tab10 colormap for colors (one color per pairing type)
    colors = plt.cm.tab10(np.linspace(0, 1, len(pairing_types)))

    stats_text = ""

    for idx, pairing_type in enumerate(pairing_types):
        try:
            steering_sims = process_pairing(dataset_name, pairing_type, device=device)
        except Exception as e:
            logging.error(f"Error processing pairing type '{pairing_type}' for dataset '{dataset_name}': {str(e)}")
            continue

        # Plot histogram of cosine similarities for the current pairing type
        ax.hist(
            steering_sims,
            bins=20,
            density=True,
            alpha=0.6,
            color=colors[idx],
            label=f"{pairing_type} (mean: {np.mean(steering_sims):.3f})",
            histtype='step',
            linewidth=2
        )

        # Append statistics for display
        stats_text += f"{pairing_type}: mean = {np.mean(steering_sims):.3f}\n"

    # Add the stats text to the plot
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(f"Steering Vector Cosine Similarities\nDataset: {dataset_name}")
    ax.legend()
    ax.set_xlim(-1, 1)


def main() -> None:
    # Get device using the provided utility function
    device = utils.get_device()

    # Get DATA_PATH and construct plots directory
    DATA_PATH = utils.get_path('DATA_PATH')
    PLOTS_PATH = os.path.join(DATA_PATH, "0_paper_plots")

    # Loop over all datasets in all_datasets_figure_13
    datasets = all_datasets_figure_13

    # Create one subplot per dataset
    fig, axs = plt.subplots(len(datasets), 1, figsize=(15, 5 * len(datasets)))
    fig.suptitle('Steering Vector Cosine Similarity Distributions Across Datasets', fontsize=16, y=0.95)

    # Ensure axs is iterable even if there's only one subplot
    if len(datasets) == 1:
        axs = [axs]

    # Process each dataset
    for idx, dataset_name in enumerate(datasets):
        logging.info(f"Processing dataset: {dataset_name}")
        try:
            plot_distributions_for_dataset(dataset_name, axs[idx], device=device)
        except Exception as e:
            logging.error(f"Error processing dataset '{dataset_name}': {str(e)}")
            continue

    plt.tight_layout()
    output_path = os.path.join(PLOTS_PATH, "cosine_similarity_steering_for_200_samples.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
