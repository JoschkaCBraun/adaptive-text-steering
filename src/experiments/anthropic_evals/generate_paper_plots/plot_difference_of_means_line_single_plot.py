"""
plot_kappa_distribution_for_steered_and_non_steered_activations.py
"""

import os
import pickle
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import torch

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


def apply_steering_vector(activations: dict, steering_vector: torch.Tensor, 
                          layer: int, factor: float) -> dict:
    # Only steer the negative activations; leave the positive activations untouched.
    result = {
         'positive': {},
         'negative': {}
     }
    result["negative"][layer] = [
        act.to('cpu') + factor * steering_vector.to('cpu')
        for act in activations["negative"][layer]
    ]
    result["positive"][layer] = activations["positive"][layer]
    return result


def plot_two_distributions_with_shared_bins(
        pos_proj_orig: np.ndarray, neg_proj_orig: np.ndarray,
        pos_proj_not_steered: np.ndarray, neg_proj_steered: np.ndarray,
        axes: List[plt.Axes]) -> None:
    """
    Plot two distributions with shared bins and x-axis range.
    Left plot: original (not steered) positive (green) and negative (red).
    Right plot: not steered positive (green), steered negative (blue), 
    and also the original negative distribution (grey).
    """
    global_min = min(
        np.min(pos_proj_orig), np.min(neg_proj_orig),
        np.min(pos_proj_not_steered), np.min(neg_proj_steered)
    )
    global_max = max(
        np.max(pos_proj_orig), np.max(neg_proj_orig),
        np.max(pos_proj_not_steered), np.max(neg_proj_steered)
    )
    bins = np.linspace(global_min, global_max, 50)

    # We'll define the data for each subplot.
    # Left subplot: "not steered" (pos=green, neg=red)
    # Right subplot: "steered neg" (pos=green, neg=blue), plus original neg (grey)
    distributions = [
        {
            'pos': pos_proj_orig,
            'neg': neg_proj_orig,
            'ax': axes[0],
            'label': 'not steered',
            'plot_grey': False
        },
        {
            'pos': pos_proj_not_steered,
            'neg': neg_proj_steered,
            'ax': axes[1],
            'label': 'steered neg',
            'plot_grey': True
        }
    ]

    for dist in distributions:
        pos_proj = dist['pos']
        neg_proj = dist['neg']
        ax = dist['ax']
        label = dist['label']

        # Plot positive distribution in green
        ax.hist(pos_proj, bins=bins, alpha=0.5, color='green', density=True,
                label='pos', rwidth=0.8)

        # If it's the left subplot, negative is not steered => red
        # If it's the right subplot, negative is steered => blue (+ original in grey)
        if label == 'steered neg':
            ax.hist(neg_proj, bins=bins, alpha=0.5, color='red', density=True,
                    label='neg (steered)', rwidth=0.8)
            # Also overlay the original negative distribution in grey
            ax.hist(neg_proj_orig, bins=bins, alpha=0.5, color='grey', density=True,
                    label='neg (orig)', rwidth=0.8)
        else:
            ax.hist(neg_proj, bins=bins, alpha=0.5, color='red', density=True,
                    label='neg', rwidth=0.8)

        ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=8, weight="bold")
        # Only add legend for the right subplot to avoid duplicates.
        # if label == "steered neg":
        #     legend = ax.legend(fontsize=6, loc='upper right', ncol=1)
        #     legend.get_frame().set_alpha(0.5)

        # ax.set_xlim(-6, 6)
        ax.set_xlim(global_min, global_max)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.set_xlabel("difference-of-means line", fontsize=9)
        ax.tick_params(axis='both', labelsize=6)

    
def main() -> None:
    # Only use indexes 0 and 33.
    dataset_idxs = [0, 33]

    # Constants
    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 500
    TARGET_LAYER = 13

    # Setup paths and device
    device = torch.device('cpu')  # Force CPU for all operations
    DATA_PATH = utils.get_path('DATA_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(DATA_PATH, "0_paper_plots")
    os.makedirs(plots_path, exist_ok=True)
    
    # Process each dataset separately.
    for dataset_idx in dataset_idxs:
        # Determine the dataset name from the list.
        dataset = all_datasets_with_index_and_fraction_anti_steerable_figure_13[dataset_idx][0]
        print(f"\nProcessing dataset: {dataset}")
        
        # Load activations.
        file_name = f"{MODEL_NAME}_activations_{dataset}_for_{NUM_SAMPLES}_samples_last.pkl"
        file_path = os.path.join(ACTIVATIONS_PATH, dataset, file_name)
        activations = load_activations(file_path, NUM_SAMPLES, device)
        
        # 1. Compute original (non-steered) kappa distributions.
        print("Computing original kappa distributions...")
        kappas_dict_pos = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=activations['positive'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        kappas_dict_neg = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=activations['negative'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        pos_proj = torch.tensor(kappas_dict_pos[TARGET_LAYER]).numpy()
        neg_proj = torch.tensor(kappas_dict_neg[TARGET_LAYER]).numpy()
        
        # 2. Compute CAA steering vector and apply steering.
        print("Computing CAA steering vector...")
        caa_steering_vector = sv.compute_contrastive_steering_vector_dict(
            positive_activations=activations['positive'],
            negative_activations=activations['negative'],
            device=device
        )[TARGET_LAYER]
        print("Applying CAA steering...")
        caa_steered_activations = {
            'positive': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, -1.0)['positive'],
            'negative': apply_steering_vector(activations, caa_steering_vector, TARGET_LAYER, 1.0)['negative']
        }
        print("Computing CAA steered kappa distributions...")
        caa_kappas_pos = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=caa_steered_activations['positive'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        caa_kappas_neg = sv.calculate_feature_expressions_kappa_dict(
            activations_dict=caa_steered_activations['negative'],
            positive_activations_dict=activations['positive'],
            negative_activations_dict=activations['negative'],
            device=device
        )
        caa_pos_proj = torch.tensor(caa_kappas_pos[TARGET_LAYER]).numpy()
        caa_neg_proj = torch.tensor(caa_kappas_neg[TARGET_LAYER]).numpy()
        
        # Create a separate figure for this dataset.
        # Set figure size to (width, height) = (5, 1.5)
        fig, axes = plt.subplots(1, 2, figsize=(6, 1.1))
        plot_two_distributions_with_shared_bins(
            pos_proj, neg_proj,
            caa_pos_proj, caa_neg_proj,
            axes
        )
        axes[0].set_ylabel("density", fontsize=9)
        
        # Place the dataset header (suptitle) with extra space.
        # Here, y=0.98 places the title near the top and we adjust the top margin to 0.7
        fig.suptitle(f"{dataset} dataset ({NUM_SAMPLES} samples, layer = 13)", fontsize=10, y=0.99)
        # Increase horizontal spacing (wspace) between plots and leave extra room for the header.
        fig.subplots_adjust(top=0.80, wspace=0.6)

        left_ax_pos = axes[0].get_position()
        right_ax_pos = axes[1].get_position()
        offset_y = 0.2  # moves the arrow up
        shorten_x = 0.03  # shortens the arrow's horizontal span
        shift_left = 0.05  # amount to shift the arrow to the left
        axes[0].annotate(
            '',
            xy=(right_ax_pos.x0 - shorten_x - shift_left, right_ax_pos.y0 + right_ax_pos.height / 2 + offset_y),
            xytext=(left_ax_pos.x1 + shorten_x - shift_left, left_ax_pos.y0 + left_ax_pos.height / 2 + offset_y),
            xycoords='figure fraction',
            arrowprops=dict(arrowstyle='->', lw=2, color='black')
        )

        # Compute the midpoint of the arrow and add a label on top:
        mid_x = ((left_ax_pos.x1 + shorten_x - shift_left) + (right_ax_pos.x0 - shorten_x - shift_left)) / 2 + 0.03
        mid_y = ((left_ax_pos.y0 + left_ax_pos.height / 2 + offset_y) + (right_ax_pos.y0 + right_ax_pos.height / 2 + offset_y)) / 2 - 0.25

        fig.text(mid_x, mid_y, "apply\nsteering", ha='center', va='bottom', fontsize=8, weight='bold')


        output_file = os.path.join(
            plots_path, 
            f'difference_of_means_line_projection_{dataset}.pdf'
        )
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Successfully saved plot for dataset {dataset} to: {output_file}")
        plt.close()
    
    print("Done!")


if __name__ == "__main__":
    main()
