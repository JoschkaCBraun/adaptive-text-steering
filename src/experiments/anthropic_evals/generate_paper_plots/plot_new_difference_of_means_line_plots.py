import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

import matplotlib.patches as mpatches

# Adjust to match your codebase
from src.utils.dataset_names import all_datasets_with_index_and_fraction_anti_steerable_figure_13
import src.utils.compute_steering_vectors as sv
import src.utils as utils


def load_activations(file_path: str, num_samples: int, device: torch.device) -> dict:
    """Load and prepare activations for kappa calculation."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    result = {
        'positive': {},
        'negative': {}
    }
    
    for class_type in ['positive', 'negative']:
        for layer in data[class_type].keys():
            layer_tensors = [tensor.to(device) for tensor in data[class_type][layer][:num_samples]]
            result[class_type][int(layer)] = layer_tensors
            
    return result


def apply_steering_vector(activations: dict, steering_vector: torch.Tensor, layer: int, factor: float) -> dict:
    """Apply a steering vector to negative activations only, leaving positives untouched."""
    result = {'positive': {}, 'negative': {}}
    result["negative"][layer] = [
        act.to('cpu') + factor * steering_vector.to('cpu')
        for act in activations["negative"][layer]
    ]
    # Positive remains unchanged
    result["positive"][layer] = activations["positive"][layer]
    return result


def plot_two_distributions(
    pos_proj_orig: np.ndarray, 
    neg_proj_orig: np.ndarray,
    pos_proj_steered: np.ndarray, 
    neg_proj_steered: np.ndarray,
    ax_left: plt.Axes, 
    ax_right: plt.Axes,
    row_idx: int
) -> None:
    """
    Plot pair of subplots for one dataset row:
     - Left subplot: original (not steered) pos (green) & neg (red).
     - Right subplot: pos (green) & neg steered (red hatch).
    """
    num_bins = 25
    
    # --- LEFT subplot: original (not steered) ---
    ax_left.hist(
        pos_proj_orig, bins=num_bins, alpha=0.5, color='green', density=True,
        label='positive', rwidth=0.8
    )
    ax_left.hist(
        neg_proj_orig, bins=num_bins, alpha=0.5, color='red', density=True,
        label='negative', rwidth=0.8
    )
    ax_left.text(
        0.05, 0.85, "not steered", transform=ax_left.transAxes,
        fontsize=8, weight="bold"
    )

    if row_idx == 1:  # second (bottom) row
        ax_left.set_xlabel("difference-of-means line", fontsize=9)
    ax_left.set_ylabel("density", fontsize=9)
    ax_left.tick_params(axis='both', labelsize=7)

    # --- RIGHT subplot: steered ---
    ax_right.hist(
        pos_proj_steered, bins=num_bins, alpha=0.5, color='green', density=True,
        label='positive', rwidth=0.8
    )
    # Use red + hatch pattern for negative steered
    ax_right.hist(
        neg_proj_steered, bins=num_bins, density=True, facecolor='red',
        alpha=0.6, hatch='+++', label='negative steered', rwidth=0.8
    )
    ax_right.text(
        0.05, 0.85, "steered", transform=ax_right.transAxes,
        fontsize=8, weight="bold"
    )

    if row_idx == 1:  # bottom row
        ax_right.set_xlabel("difference-of-means line", fontsize=9)

    ax_right.tick_params(axis='both', labelsize=7)


def main():
    # Example dataset indexes
    dataset_idxs = [0, 33]

    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 500
    TARGET_LAYER = 13

    device = torch.device('cpu')
    DATA_PATH = utils.get_path('DATA_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(DATA_PATH, "0_paper_plots")
    os.makedirs(plots_path, exist_ok=True)

    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 2.8))

    # Adjust spacing:
    # - top=0.95 leaves room for the suptitle at y=0.98
    # - bottom=0.1 is a bit more space below
    # - wspace=0.6 is moderate gap between subplots
    # - hspace=0.4 is some vertical space between rows
    plt.subplots_adjust(top=0.83, bottom=0.12, left=0.08, right=0.95, wspace=0.22, hspace=0.22)

    # ---- Suptitle (header) near the very top
    fig.suptitle(
        "Activations projected on difference-of-means line",
        fontsize=11, 
        y=1.05
    )

    # ---- Legend below header, above plots
    # 4 columns, semi-transparent background
    alpha_patches = 0.6
    red_patch = mpatches.Patch(color='red', label='negative activations', alpha=alpha_patches)
    green_patch = mpatches.Patch(color='green', label='positive activations', alpha=alpha_patches)
    negative_steered_patch = mpatches.Patch(
        facecolor='red', hatch='+++', alpha=alpha_patches, label='negative activations after steering'
    )

    fig.legend(
        handles=[red_patch, green_patch, negative_steered_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),  # just below the suptitle
        ncol=3,
        frameon=True,
        facecolor='white',
        framealpha=0.5,
        fontsize=8
    )

    for row_idx, dataset_idx in enumerate(dataset_idxs):
        dataset = all_datasets_with_index_and_fraction_anti_steerable_figure_13[dataset_idx][0]
        print(f"Processing dataset: {dataset}")

        # Load activations
        file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
        file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
        activations = load_activations(file_path, NUM_SAMPLES, device)

        # Compute original (non-steered) kappa distributions
        kappas_pos = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=activations['positive'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        kappas_neg = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=activations['negative'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        pos_proj_orig = np.array(kappas_pos[TARGET_LAYER])
        neg_proj_orig = np.array(kappas_neg[TARGET_LAYER])

        # Compute steering vector, apply
        steering_vecs = sv.compute_contrastive_steering_vector_dict(
            positive_activations=activations['positive'],
            negative_activations=activations['negative'],
            device=device
        )
        steering_vec = steering_vecs[TARGET_LAYER]

        steered_acts = {
            'positive': apply_steering_vector(activations, steering_vec, TARGET_LAYER, -1.0)['positive'],
            'negative': apply_steering_vector(activations, steering_vec, TARGET_LAYER, 1.0)['negative']
        }

        steered_pos_kappa = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=steered_acts['positive'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        steered_neg_kappa = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=steered_acts['negative'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )

        pos_proj_steered = np.array(steered_pos_kappa[TARGET_LAYER])
        neg_proj_steered = np.array(steered_neg_kappa[TARGET_LAYER])

        # Axes for this row
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        # Plot the two-subplot distribution
        plot_two_distributions(
            pos_proj_orig, 
            neg_proj_orig,
            pos_proj_steered, 
            neg_proj_steered,
            ax_left, 
            ax_right,
            row_idx
        )

        # x-limits
        top_xlim = 1.8
        bottom_xlim = 6.5
        if row_idx == 0:
            ax_left.set_xlim(-top_xlim, top_xlim)
            ax_right.set_xlim(-top_xlim, top_xlim)
        else:
            ax_left.set_xlim(-bottom_xlim, bottom_xlim)
            ax_right.set_xlim(-bottom_xlim, bottom_xlim)

        # Move dataset name to the center above the row (between subplots)
        left_ax_pos = ax_left.get_position()
        right_ax_pos = ax_right.get_position()
        mid_x = (left_ax_pos.x0 + right_ax_pos.x1) / 2
        # place the text slightly above the subplots
        name_y = left_ax_pos.y0 + left_ax_pos.height + 0.02
        fig.text(
            mid_x, 
            name_y, 
            f"{dataset} dataset", 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            weight='bold'
        )

        # Add arrow from left to right
        # We'll move it up (offset_y = +0.05 from center)
        arrow_offset_y = -0.02
        arrow_offset_x = 0.00
        shorten_x = -0.02
        arrow_start_x = left_ax_pos.x1 + shorten_x + arrow_offset_x
        arrow_end_x = right_ax_pos.x0 - shorten_x + arrow_offset_x
        arrow_y = left_ax_pos.y0 + left_ax_pos.height / 2 + arrow_offset_y

        ax_left.annotate(
            '',  # no text, just the arrow
            xy=(arrow_end_x, arrow_y),
            xytext=(arrow_start_x, arrow_y),
            xycoords='figure fraction',
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle='->', lw=2, color='black')
        )

        # Add "apply steering" text near the arrow's midpoint
        arrow_mid_x = (arrow_start_x + arrow_end_x) / 2
        ax_left.annotate(
            "apply  \nsteering  ",
            xy=(arrow_mid_x, arrow_y),
            xycoords='figure fraction',
            textcoords='offset points',
            xytext=(0, 20),  # shift text down from arrow
            ha='center', 
            va='top', 
            fontsize=8, 
            weight='bold'
        )

    output_file = os.path.join(plots_path, "combined_difference_of_means_line_projection.pdf")
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved combined plot to {output_file}")


if __name__ == "__main__":
    main()
