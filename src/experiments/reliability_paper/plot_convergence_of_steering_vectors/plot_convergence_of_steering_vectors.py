"""
plot_convergence_of_steering_vectors.py
"""
import os
import pickle
import random
from typing import Dict, List, Tuple, NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

import src.utils as utils
from src.utils import compute_contrastive_steering_vector, compute_many_against_one_cosine_similarity
from src.utils.dataset_names import all_datasets_figure_13_tan_et_al

class PromptPair(NamedTuple):
    """Named tuple to store the names of the positive and negative prompts."""
    name: str
    positive: str
    negative: str

def load_dataset_results(activations_path: str, dataset_name: str, model_name: str, 
                         num_samples: int, prompt_type: str) -> List:
    """Load and extract relevant activations from the stored pickle file for a specific prompt type."""
    file_path = os.path.join(
        activations_path,
        dataset_name,
        f"{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    
    print(f"\nProcessing dataset: {dataset_name}, prompt type: {prompt_type}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract results for the specified prompt type
    results = data['results'][prompt_type]
    
    # Create activation list
    activations = [result['activation_layer_13'].clone().detach() 
                  for result in results]
    
    return activations

def compute_convergence_metrics(
    positive_activations: List,
    negative_activations: List,
    sample_sizes: List[int],
    num_trials: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute convergence metrics for different sample sizes."""
    full_steering_vector = compute_contrastive_steering_vector(
        positive_activations, negative_activations, device)
    
    mean_similarities = []
    std_similarities = []
    
    for size in sample_sizes:
        trial_similarities = []
        
        for _ in range(num_trials):
            indices = random.sample(range(len(positive_activations)), size)
            
            pos_subset = [positive_activations[i] for i in indices]
            neg_subset = [negative_activations[i] for i in indices]
            
            subset_steering_vector = compute_contrastive_steering_vector(
                pos_subset, neg_subset, device)
            
            similarity = compute_many_against_one_cosine_similarity(
                [subset_steering_vector], full_steering_vector, device)
            
            trial_similarities.append(similarity)
        
        mean_similarities.append(np.mean(trial_similarities))
        std_similarities.append(np.std(trial_similarities))
    
    return np.array(mean_similarities), np.array(std_similarities)

def plot_convergence_subplots(
    all_results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    sample_sizes: List[int],
    output_path: str,
    prompt_pairs: List[PromptPair],
    num_trials: int = 5  # Added parameter for number of trials
) -> None:
    """Create subplots for each prompt type pair."""
    num_pairs = len(prompt_pairs)
    fig, axes = plt.subplots(num_pairs, 1, figsize=(15, 6*num_pairs))
    
    for idx, (ax, prompt_pair) in enumerate(zip(axes, prompt_pairs)):
        # Create subplot and legend area
        divider = make_axes_locatable(ax)
        legend_ax = divider.append_axes("right", size="20%", pad=0.1)
        
        # Plot convergence lines for each dataset
        for dataset_idx, (dataset_name, results) in enumerate(all_results.items()):
            means, stds = results[prompt_pair.name]
            color = plt.cm.viridis(dataset_idx / len(all_results))
            ax.errorbar(sample_sizes, means, yerr=stds, color=color, capsize=3)
        
        # Main plot formatting
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'Steering Vector Convergence\nPrompt type: {prompt_pair.name} (n={num_trials} trials)')
        ax.grid(True)
        
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        
        for dataset_idx, dataset_name in enumerate(all_results.keys()):
            color = plt.cm.viridis(dataset_idx / len(all_results))
            rect = plt.Rectangle((0, 1 - (dataset_idx + 1) / len(all_results)), 
                               0.2, 
                               1/len(all_results), 
                               facecolor=color)
            legend_ax.add_patch(rect)
            legend_ax.text(0.3, 
                          1 - (dataset_idx + 0.5) / len(all_results),
                          f"{dataset_idx+1}. {dataset_name}",
                          va='center',
                          fontsize=8)
        
        legend_ax.set_title('Datasets')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main() -> None:
    DISK_PATH = os.getenv('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    PROJECT_PATH = utils.get_path('PROJECT_PATH')
    PLOTS_PATH = os.path.join(PROJECT_PATH, "data", "reliability_paper_data", "plots")
    MODEL_NAME = 'llama2_7b_chat'
    TOTAL_SAMPLES = 200
    STEP_SIZE = 20
    NUM_TRIALS = 2
    NUM_DATASETS = 5
    PROMPT_PAIRS = [
        PromptPair("Prefilled", "matching_prefilled", "non_matching_prefilled"),
        PromptPair("Instruction", "matching_instruction", "non_matching_instruction"),
        PromptPair("Instruction+Prefilled", "matching_instruction_prefilled", "non_matching_instruction_prefilled"),
        PromptPair("Instruction+FewShot", "matching_instruction_few_shot", "non_matching_instruction_few_shot"),
        PromptPair("Instruction+FewShot+Prefilled", "matching_instruction_few_shot_prefilled", "non_matching_instruction_few_shot_prefilled")
    ]
    
    device = utils.get_device()
    datasets = all_datasets_figure_13_tan_et_al # [:NUM_DATASETS]
    sample_sizes = list(range(STEP_SIZE, TOTAL_SAMPLES + STEP_SIZE, STEP_SIZE))
    
    # Dictionary to store results for all datasets and prompt types
    all_results = {}
    
    for dataset_name in datasets:
        try:
            dataset_results = {}
            
            for prompt_pair in PROMPT_PAIRS:
                # Load activations for both positive and negative samples
                pos_activations = load_dataset_results(
                    ACTIVATIONS_PATH, dataset_name, MODEL_NAME, TOTAL_SAMPLES, prompt_pair.positive)
                neg_activations = load_dataset_results(
                    ACTIVATIONS_PATH, dataset_name, MODEL_NAME, TOTAL_SAMPLES, prompt_pair.negative)
                
                # Compute metrics
                means, stds = compute_convergence_metrics(
                    pos_activations, neg_activations, sample_sizes, NUM_TRIALS, device)
                
                dataset_results[prompt_pair.name] = (means, stds)
            
            all_results[dataset_name] = dataset_results
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
    
    output_path = os.path.join(PLOTS_PATH, "steering_vector_convergence_all_prompt_types.pdf")
    plot_convergence_subplots(all_results, sample_sizes, output_path, PROMPT_PAIRS)


if __name__ == "__main__":
    main()