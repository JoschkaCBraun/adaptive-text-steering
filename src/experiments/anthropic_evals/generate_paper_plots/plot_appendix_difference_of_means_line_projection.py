"""
plot_kappa_distribution_for_steered_and_non_steered_activations.py
"""

import os
import pickle
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import torch

# We'll use PyPDF2 to merge the individual PDFs.
from PyPDF2 import PdfMerger

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
    """Apply steering vector to activations with given factor."""
    result = {
        'positive': {},
        'negative': {}
    }
    
    for key in ['positive', 'negative']:
        result[key][layer] = [
            act.to('cpu') + factor * steering_vector.to('cpu')
            for act in activations[key][layer]
        ]
    return result


def plot_two_distributions_with_shared_bins(
        pos_proj_orig: np.ndarray, neg_proj_orig: np.ndarray,
        pos_proj_caa: np.ndarray, neg_proj_caa: np.ndarray,
        axes: List[plt.Axes]) -> None:
    """
    Plot two distributions (non-steered and steered) with shared bins and x-axis range.
    This function does not add dataset-specific headers.
    """
    # Determine a common range for the histograms.
    global_min = min(np.min(pos_proj_orig), np.min(neg_proj_orig),
                     np.min(pos_proj_caa), np.min(neg_proj_caa))
    global_max = max(np.max(pos_proj_orig), np.max(neg_proj_orig),
                     np.max(pos_proj_caa), np.max(neg_proj_caa))
    bins = np.linspace(global_min, global_max, 50)
    
    # Define the two sets of distributions and corresponding labels.
    distributions = [
        (pos_proj_orig, neg_proj_orig),
        (pos_proj_caa, neg_proj_caa)
    ]
    labels = ["not steered", "steered"]
    
    for i, ((pos_proj, neg_proj), label, ax) in enumerate(zip(distributions, labels, axes)):
        # Plot histograms for negative and positive projections.
        ax.hist(pos_proj, bins=bins, alpha=0.5, color='red', density=True,
                label='pos', rwidth=0.8)
        ax.hist(neg_proj, bins=bins, alpha=0.5, color='blue', density=True,
                label='neg', rwidth=0.8)

        # Move the label a bit upward (y coordinate set to 0.85).
        ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=8, weight="bold")
        # Add legend only on the right subplot (i == 1) and make it semi-transparent.
        if i == 1:
            legend = ax.legend(fontsize=6, loc='upper right', ncol=2)
            legend.get_frame().set_alpha(0.5)
        ax.set_xlim(global_min, global_max)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))


def main() -> None:
    # choose datasets to process
    dataset_idxs = range(36)

    # Constants
    MODEL_NAME = 'llama2_7b_chat'
    NUM_SAMPLES = 500
    TARGET_LAYER = 13

    # Setup paths and device.
    device = torch.device('cpu')  # Force CPU for all operations
    DATA_PATH = utils.get_path('DATA_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    plots_path = os.path.join(DATA_PATH, "0_paper_plots")
    os.makedirs(plots_path, exist_ok=True)
    
    # Create a temporary directory to hold individual plot PDFs.
    temp_plots_dir = os.path.join(plots_path, "temp_plots")
    os.makedirs(temp_plots_dir, exist_ok=True)
    
    # List to hold the file paths of the temporary PDFs.
    temp_plot_paths = []

    # Process each dataset separately.
    for i, dataset_idx in enumerate(dataset_idxs):
        # Determine the dataset name.
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
        
        # Create a figure for this dataset exactly as before.
        fig, axes = plt.subplots(1, 2, figsize=(6, 1.1))
        plot_two_distributions_with_shared_bins(
            pos_proj, neg_proj,
            caa_pos_proj, caa_neg_proj,
            axes
        )
        axes[0].set_ylabel("density", fontsize=10)
        
        # Place the dataset header as a suptitle.
        fig.suptitle(f"{dataset} dataset (n = {NUM_SAMPLES})", fontsize=10, y=0.99)
        fig.subplots_adjust(top=0.80, wspace=0.6)
        
        # Draw an arrow between the two subplots.
        left_ax_pos = axes[0].get_position()
        right_ax_pos = axes[1].get_position()
        offset_y = -0.02   # moves the arrow up a bit
        shorten_x = 0.02   # shortens the arrow's horizontal span
        shift_left = 0.02  # amount to shift the arrow to the left
        axes[0].annotate(
            '',
            xy=(right_ax_pos.x0 - shorten_x - shift_left, right_ax_pos.y0 + right_ax_pos.height / 2 + offset_y),
            xytext=(left_ax_pos.x1 + shorten_x - shift_left, left_ax_pos.y0 + left_ax_pos.height / 2 + offset_y),
            xycoords='figure fraction',
            arrowprops=dict(arrowstyle='->', lw=2, color='black')
        )
        
        # Compute the midpoint of the arrow and add a label on top.
        mid_x = ((left_ax_pos.x1 + shorten_x - shift_left) + (right_ax_pos.x0 - shorten_x - shift_left)) / 2 + 0.02
        mid_y = ((left_ax_pos.y0 + left_ax_pos.height / 2 + offset_y) + (right_ax_pos.y0 + right_ax_pos.height / 2 + offset_y)) / 2 - 0.14
        fig.text(mid_x, mid_y, "apply\nsteering", ha='center', va='bottom', fontsize=8, weight='bold')
        
        # Save the individual figure to a PDF file exactly as before.
        temp_file = os.path.join(temp_plots_dir, f"temp_plot_{i}.pdf")
        fig.savefig(temp_file, bbox_inches='tight')
        temp_plot_paths.append(temp_file)
        print(f"Saved temporary plot for dataset {dataset} at {temp_file}")
        plt.close(fig)
    
    # Now merge all individual PDFs into one multi-page PDF.
    final_output_file = os.path.join(
        plots_path, 
        f'kappa_distribution_polished_caa_layer_{TARGET_LAYER}_{MODEL_NAME}_multiple_datasets.pdf'
    )
    
    merger = PdfMerger()
    for pdf in temp_plot_paths:
        merger.append(pdf)
    merger.write(final_output_file)
    merger.close()
    
    print(f"\nSuccessfully saved multi-dataset plot to: {final_output_file}")
    
    # Optionally, clean up the temporary PDF files.
    for pdf in temp_plot_paths:
        os.remove(pdf)
    os.rmdir(temp_plots_dir)
    print("Cleaned up temporary plot files.")


if __name__ == "__main__":
    main()
