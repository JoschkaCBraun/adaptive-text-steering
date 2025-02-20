import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches
from src.utils import get_path

# Constants
SUBSET_SIZE_K = 40
PROMPT_TYPES = ['instruction', '5-shot', 'instruction_5-shot']
SUBSET_TYPES = ['full', 'top_k', 'median_k', 'bottom_k', 'random_k']
SUBSET_NAMES = ['Full', 'Top-20', 'Median-20', 'Bottom-20', 'Random-20']

def load_baseline_differences(baseline_dir: str, dataset_name: str) -> np.ndarray:
    """Load and compute baseline differences for a given dataset."""
    baseline_file = f"{dataset_name}_steered_and_baseline_logits_250_samples_36_datasets.h5"
    baseline_path = os.path.join(baseline_dir, baseline_file)
    
    try:
        with h5py.File(baseline_path, 'r') as f:
            if 'no_steering' not in f:
                raise KeyError(f"No baseline data found for {dataset_name}")
                
            baseline_group = f['no_steering']
            baseline_matching = np.array(baseline_group['matching_logits'])
            baseline_non_matching = np.array(baseline_group['non_matching_logits'])
            return baseline_matching - baseline_non_matching
            
    except Exception as e:
        print(f"Error loading baseline for {dataset_name}: {str(e)}")
        return None

def load_and_process_h5_dataset(file_path: str, baseline_diffs: np.ndarray = None) -> Dict:
    """Load and process single h5 file, computing per-subset steering effects."""
    results = {}
    
    with h5py.File(file_path, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        
        for prompt_type in PROMPT_TYPES:
            if prompt_type not in f:
                continue
                
            results[prompt_type] = {}
            prompt_group = f[prompt_type]
            
            for subset_type in SUBSET_TYPES:
                if subset_type not in prompt_group:
                    continue
                    
                subset_group = prompt_group[subset_type]
                
                # Get logits and compute differences
                matching_logits = np.array(subset_group['matching_logits'])
                non_matching_logits = np.array(subset_group['non_matching_logits'])
                
                # Compute steering effect
                steering_effects = matching_logits - non_matching_logits
                
                # Subtract baseline if provided
                if baseline_diffs is not None:
                    steering_effects = steering_effects - baseline_diffs
                
                results[prompt_type][subset_type] = {
                    'steering_effects': steering_effects.flatten().tolist()
                }
    
    return results

def calculate_dataset_averages(all_results: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Calculate steering effects across all datasets for each prompt type and subset."""
    combined_effects = {
        prompt_type: {subset_type: [] for subset_type in SUBSET_TYPES}
        for prompt_type in PROMPT_TYPES
    }
    
    for dataset_results in all_results.values():
        for prompt_type in PROMPT_TYPES:
            if prompt_type not in dataset_results:
                continue
            for subset_type in SUBSET_TYPES:
                if subset_type in dataset_results[prompt_type]:
                    combined_effects[prompt_type][subset_type].extend(
                        dataset_results[prompt_type][subset_type]['steering_effects']
                    )
    
    return combined_effects

def calculate_statistics(data: List[float]) -> Tuple[float, float]:
    """Calculate mean and anti-steerable proportion for a distribution."""
    if not data:
        return 0.0, 0.0
    mean = np.mean(data)
    anti_steerable = (np.array(data) < 0).mean() * 100
    return mean, anti_steerable

def create_violin_subplot(ax: plt.Axes, data: Dict[str, Dict[str, List[float]]], 
                        prompt_colors: np.ndarray, subset_patterns: List[str],
                        y_limits: Tuple[float, float], title: str = None) -> List[Tuple]:
    """Create a single violin subplot with given data and styling."""
    positions = []
    violin_stats = []
    current_pos = 0
    
    # Create violins with patterns
    for p_idx, prompt_type in enumerate(PROMPT_TYPES):
        for s_idx, subset_type in enumerate(SUBSET_TYPES):
            pos = current_pos + s_idx
            positions.append(pos)
            
            subset_data = data[prompt_type][subset_type]
            if subset_data:
                violin_parts = ax.violinplot(subset_data, positions=[pos], 
                                           points=250, widths=0.7,
                                           showmeans=True, showextrema=False)
                
                # Apply color and pattern
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(prompt_colors[p_idx])
                    pc.set_alpha(0.7)
                    if subset_patterns[s_idx]:
                        pc.set_hatch(subset_patterns[s_idx])
                
                violin_parts['cmeans'].set_color('black')
                violin_parts['cmeans'].set_linewidth(2)
                
                # Store statistics for legend
                mean, anti_steerable = calculate_statistics(subset_data)
                violin_stats.append((prompt_type, subset_type, mean, anti_steerable))
        
        current_pos += len(SUBSET_TYPES) + 2  # Add space between prompt types
    
    # Customize subplot
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel('Per-Sample\nSteering Effect', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1, zorder=1)
    
    # Set x-ticks in groups
    group_positions = np.arange(0, current_pos, len(SUBSET_TYPES) + 2)
    group_centers = group_positions + (len(SUBSET_TYPES) - 1) / 2
    ax.set_xticks(positions)
    ax.set_xticklabels(SUBSET_NAMES * len(PROMPT_TYPES), rotation=45, ha='right', fontsize=8)
    
    # Add prompt type labels
    for pos, prompt_type in zip(group_centers, PROMPT_TYPES):
        ax.text(pos, y_limits[0] - 0.05 * (y_limits[1] - y_limits[0]),
                prompt_type, ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Set y limits
    ax.set_ylim(y_limits)
    
    # Add statistics legend with color patches
    stats_legend = []
    legend_patches = []
    
    for stat in violin_stats:
        prompt_type, subset_type, mean, anti = stat
        p_idx = PROMPT_TYPES.index(prompt_type)
        s_idx = SUBSET_TYPES.index(subset_type)
        
        # Create patch with same color and pattern as violin
        patch = mpatches.Patch(facecolor=prompt_colors[p_idx], alpha=0.7,
                             hatch=subset_patterns[s_idx])
        legend_patches.append(patch)
        stats_legend.append(f'{mean:.2f} / {anti:.1f}%')
    
    # Create legend with color patches and stats
    ax.legend(legend_patches, stats_legend,
             loc='center left',
             bbox_to_anchor=(1.01, 0.5),
             title='mean / % anti-steerable',
             fontsize=8,
             title_fontsize=8)
    
    return violin_stats

def create_multi_dataset_plot(all_results: Dict, avg_data: Dict, output_path: str) -> None:
    """Create multi-subplot figure with average and per-dataset results."""
    num_datasets = len(all_results)
    fig_height = 4 * (num_datasets + 1)  # +1 for the average plot
    
    fig, axes = plt.subplots(num_datasets + 1, 1, figsize=(15, fig_height))
    plt.subplots_adjust(hspace=0.4)
    
    # Define shared styling
    prompt_colors = plt.cm.Set2(np.linspace(0, 1, len(PROMPT_TYPES)))
    subset_patterns = ['', '/', '\\', 'x', '.']  # '' is solid fill
    
    # Calculate global y-limits
    min_val = float('inf')
    max_val = float('-inf')
    
    # Check average data
    for prompt_data in avg_data.values():
        for subset_data in prompt_data.values():
            if subset_data:
                min_val = min(min_val, min(subset_data))
                max_val = max(max_val, max(subset_data))
    
    # Check individual dataset data
    for dataset_results in all_results.values():
        for prompt_data in dataset_results.values():
            for subset_data in prompt_data.values():
                if 'steering_effects' in subset_data:
                    min_val = min(min_val, min(subset_data['steering_effects']))
                    max_val = max(max_val, max(subset_data['steering_effects']))
    
    # Add padding to y-limits
    y_limits = (min_val + 3, max_val - 3)
    
    # Create average plot
    create_violin_subplot(axes[0], avg_data, prompt_colors, subset_patterns, y_limits,
                        'Average Across All Datasets')
    
    # Create individual dataset plots
    for idx, (dataset_name, dataset_results) in enumerate(all_results.items(), 1):
        # Convert dataset results to the same format as average data
        plot_data = {
            prompt_type: {
                subset_type: dataset_results[prompt_type][subset_type]['steering_effects']
                for subset_type in SUBSET_TYPES
                if subset_type in dataset_results[prompt_type]
            }
            for prompt_type in PROMPT_TYPES
            if prompt_type in dataset_results
        }
        
        create_violin_subplot(axes[idx], plot_data, prompt_colors, subset_patterns, y_limits,
                            f'Dataset: {dataset_name}')
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def process_and_plot_results(input_dir: str, baseline_dir: str, output_dir: str, num_datasets: int) -> None:
    """Process h5 files and create plot."""
    all_results = {}
    
    # Get all h5 files
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('_samples.h5')]
    h5_files = h5_files[:num_datasets]
    
    # Load all datasets
    for h5_file in h5_files:
        dataset_name = h5_file.split('_steered_subsets')[0]
        file_path = os.path.join(input_dir, h5_file)
        
        # Load baseline differences for this dataset
        baseline_diffs = load_baseline_differences(baseline_dir, dataset_name)
        if baseline_diffs is None:
            print(f"Skipping {dataset_name} due to missing baseline")
            continue
        
        try:
            results = load_and_process_h5_dataset(file_path, baseline_diffs)
            if results:  # Only add if we got valid results
                all_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing {h5_file}: {str(e)}")
            continue
    
    if not all_results:
        print("No valid results found")
        return
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate averages
    avg_data = calculate_dataset_averages(all_results)
    
    # Create plot
    output_path = os.path.join(output_dir, 
                              f'steering_vectors_subset_analysis_{SUBSET_SIZE_K}_selected_test.pdf')
    create_multi_dataset_plot(all_results, avg_data, output_path)

def main() -> None:
    # Setup paths using utils.get_path
    anthropic_evals_results_dir = os.path.join(get_path('DISK_PATH'), 'anthropic_evals_results')
    input_dir = os.path.join(anthropic_evals_results_dir, f'effects_of_prompt_subsets_size_{SUBSET_SIZE_K}_on_steerability')
    baseline_dir = os.path.join(anthropic_evals_results_dir, 'effects_of_prompts_on_steerability')
    output_dir = os.path.join(get_path('DATA_PATH'), '0_paper_plots')
    
    process_and_plot_results(input_dir, baseline_dir, output_dir, num_datasets=36)

if __name__ == "__main__":
    main()