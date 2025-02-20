"""
plot_selective_sample_wise_logit_difference_for_steered_outputs.py
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Constants
SUBSET_SIZE_K = 20
import matplotlib.patches as mpatches

# Constants
PROMPT_TYPES = ['instruction', '5-shot', 'instruction_5-shot']
SUBSET_TYPES = ['full', 'top_k', 'median_k', 'bottom_k', 'random_k']
SUBSET_NAMES = ['Full', 'Top-20', 'Median-20', 'Bottom-20', 'Random-20']

def load_and_process_h5_dataset(file_path: str) -> Dict:
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

def create_enhanced_violin_plot(data: Dict[str, Dict[str, List[float]]], output_path: str) -> None:
    """Create and save enhanced violin plot with patterns and dual legend."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors and patterns for prompt types and subsets
    prompt_colors = plt.cm.Set2(np.linspace(0, 1, len(PROMPT_TYPES)))
    subset_patterns = ['', '/', '\\', 'x', '.']  # '' is solid fill
    
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
    
    # Customize plot
    ax.set_title('Per-Sample Steering Impact: Effect of Training Sample Selection\n'
                 'Averaged across datasets with 250 eval samples each',
                 fontsize=14, pad=20)
    ax.set_ylabel('Per-Sample Steering Effect', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1, zorder=1)
    
    # Set x-ticks in groups
    group_positions = np.arange(0, current_pos, len(SUBSET_TYPES) + 2)
    group_centers = group_positions + (len(SUBSET_TYPES) - 1) / 2
    ax.set_xticks(positions)
    ax.set_xticklabels(SUBSET_NAMES * len(PROMPT_TYPES), rotation=45, ha='right')
    
    # Add prompt type labels
    for pos, prompt_type in zip(group_centers, PROMPT_TYPES):
        ax.text(pos, ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                prompt_type, ha='center', va='top', fontsize=12, fontweight='bold')
    
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
    
    # Position legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Create legend with color patches and stats
    ax.legend(legend_patches, stats_legend,
             loc='center left',
             bbox_to_anchor=(1.01, 0.5),
             title='mean / % anti-steerable')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_and_plot_results(input_dir: str, output_dir: str, num_datasets: int) -> None:
    """Process h5 files and create plot."""
    all_results = {}
    
    # Get all h5 files
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('_samples.h5')]
    h5_files = h5_files[:num_datasets]
    
    # Load all datasets
    for h5_file in h5_files:
        dataset_name = h5_file.split('_steered_subsets')[0]
        file_path = os.path.join(input_dir, h5_file)
        
        try:
            results = load_and_process_h5_dataset(file_path)
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
    
    # Calculate averages and create plot
    avg_data = calculate_dataset_averages(all_results)
    
    output_path = os.path.join(output_dir, f'steering_vectors_subset_analysis_{SUBSET_SIZE_K}_selected.pdf')
    create_enhanced_violin_plot(avg_data, output_path)

def main() -> None:
    # Setup paths
    input_dir = os.path.join(os.environ.get('DISK_PATH', ''), 'anthropic_evals_results',
                            f'effects_of_prompt_subsets_size_{SUBSET_SIZE_K}_on_steerability')
    output_dir = os.path.join(os.environ.get('DATA_PATH', ''), '0_paper_plots')
    
    process_and_plot_results(input_dir, output_dir, num_datasets=36)

if __name__ == "__main__":
    main()