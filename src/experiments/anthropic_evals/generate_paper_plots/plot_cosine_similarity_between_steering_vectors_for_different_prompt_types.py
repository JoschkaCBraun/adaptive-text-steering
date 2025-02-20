#!/usr/bin/env python3
"""
compare_steering_cosine_similarity.py

This script loads stored steering vectors from HDF5 files located in the
'anthropic_evals_results/effects_of_prompts_on_steerability' directory.
For each file, it computes a cosine similarity matrix comparing the steering vectors
across different prompt types (excluding the 'no_steering' group), computes an average
matrix across all datasets, and then saves all the plots into a single PDF file.
Each cell of the matrix is rendered as a discrete square with a sharp border, colored
according to its cosine similarity value and annotated with the numeric value.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import get_path

def load_steering_vectors(file_path):
    """
    Load steering vectors from a given HDF5 file, skipping groups 'metadata' and 'no_steering'.

    Parameters:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary where keys are prompt type names and values are
              the corresponding steering vectors (as NumPy arrays).
    """
    steering_vectors = {}
    with h5py.File(file_path, 'r') as f:
        for group_name in f:
            if group_name in ['metadata', 'no_steering']:
                continue  # Skip metadata and no_steering groups
            try:
                vector = f[group_name]['steering_vector'][...]
                steering_vectors[group_name] = vector
            except Exception as e:
                print(f"Warning: Could not load steering vector from group '{group_name}': {e}")
    return steering_vectors

def compute_cosine_similarity(steering_vectors):
    """
    Compute the cosine similarity matrix for the given steering vectors.

    Parameters:
        steering_vectors (dict): Dictionary of prompt types to steering vectors.

    Returns:
        tuple: (list of prompt type names, similarity matrix as a 2D NumPy array)
    """
    prompt_types = list(steering_vectors.keys())
    # Stack vectors into a 2D array (each row corresponds to one prompt type)
    vectors = np.array([steering_vectors[pt] for pt in prompt_types])
    
    # Normalize each vector (adding a small epsilon to avoid division by zero).
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    normalized_vectors = vectors / norms
    
    # Compute cosine similarity as the dot product between normalized vectors.
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    
    return prompt_types, similarity_matrix

def plot_similarity_matrix(prompt_types, similarity_matrix, title="Cosine Similarity Matrix"):
    """
    Plot the cosine similarity matrix as a discrete grid of squares with clear borders.
    Each square is annotated with its numeric value.
    
    Color coding:
      - -1 (bad) is rendered in red,
      -  0 (neutral) is rendered in yellow,
      -  1 (good) is rendered in green.
      
    Discrete color steps are used and gridlines separate each cell.
    
    Parameters:
        prompt_types (list): List of prompt type names.
        similarity_matrix (np.array): 2D array containing cosine similarities.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a discrete version of RdYlGn with 10 distinct colors.
    cmap = plt.get_cmap('RdYlGn', 10)
    im = ax.imshow(similarity_matrix, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
    
    # Create grid lines by setting minor ticks and drawing grid
    ax.set_xticks(np.arange(len(prompt_types)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(prompt_types)+1)-0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Set major ticks with the prompt type names.
    ax.set_xticks(np.arange(len(prompt_types)))
    pretty_names = ['prefilled', 'instruction', '5-shot', "prefilled\ninstruction",
                "prefilled\n5-shot", "instruction\n5-shot", "prefilled\ninstruction\n5-shot"]
    ax.set_xticklabels(pretty_names)
    ax.set_yticks(np.arange(len(prompt_types)))
    ax.set_yticklabels(pretty_names)
    
    ax.set_title(title)
    fig.tight_layout()
    fig.set_dpi(300)
    
    # Annotate each cell with its numeric value.
    norm = plt.Normalize(vmin=-1, vmax=1)
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            value = similarity_matrix[i, j]
            # Get the background color for the current cell from the discrete colormap.
            rgb = cmap(norm(value))[:3]
            # Compute brightness (luminance) for contrast.
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = 'white' if luminance < 0.5 else 'black'
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)
    
    # Add a colorbar with discrete tick marks.
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('Cosine Similarity')
    
    return fig

def main() -> None:
    # Define the directory containing the HDF5 files.
    evals_dir = os.path.join(get_path('DISK_PATH'), 'anthropic_evals_activations_500')
    if not os.path.exists(evals_dir):
        print(f"Directory '{evals_dir}' does not exist.")
        return

    # Define the output directory and PDF file path.
    output_dir = os.path.join(get_path('DATA_PATH'), '0_paper_plots')
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "cosine_similarity_steering_vectors.pdf")
    
    # List all HDF5 files (.h5 extension) in the evals directory.
    h5_files = [os.path.join(evals_dir, f) for f in os.listdir(evals_dir) if f.endswith('.h5')]
    if not h5_files:
        print("No HDF5 files found in the directory.")
        return

    # Prepare to store similarity matrices for averaging.
    all_matrices = []
    common_prompt_types = None
    file_plots = []  # To store (title, similarity_matrix) for each file

    for file_path in h5_files:
        steering_vectors = load_steering_vectors(file_path)
        if not steering_vectors:
            print(f"No steering vectors found in file {file_path}.")
            continue

        prompt_types, sim_matrix = compute_cosine_similarity(steering_vectors)
        if common_prompt_types is None:
            common_prompt_types = prompt_types
        else:
            # Optional: Warn if prompt types differ across files.
            if set(common_prompt_types) != set(prompt_types):
                print(f"Warning: Prompt types differ in file {file_path}. Using common set from the first file.")
        all_matrices.append(sim_matrix)
        file_title = f"Cosine Similarity Matrix for {os.path.basename(file_path)}"
        file_plots.append((file_title, sim_matrix))

    # Create the PDF file and save plots.
    with PdfPages(pdf_path) as pdf:
        # 1. Plot the average cosine similarity matrix (across all datasets) as the first page.
        if all_matrices:
            avg_matrix = np.mean(all_matrices, axis=0)
            avg_title = "Average Cosine Similarity Matrix (across all datasets)"
            fig_avg = plot_similarity_matrix(common_prompt_types, avg_matrix, title=avg_title)
            pdf.savefig(fig_avg, bbox_inches='tight')
            plt.close(fig_avg)
            print("Saved average cosine similarity matrix.")
        
        print(f"All plots have been saved to {pdf_path}")

if __name__ == "__main__":
    main()
