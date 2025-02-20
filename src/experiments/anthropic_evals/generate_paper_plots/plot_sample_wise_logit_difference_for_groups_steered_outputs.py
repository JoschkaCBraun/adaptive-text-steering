"""
plot_sample_wise_logit_difference_for_steered_outputs.py
Analyzes h5 files and creates visualization of steering vector effects.
Computes per-sample steering effect relative to no_steering baseline.
Creates two separate plots for two groups of dataset names.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import src.utils as utils

def load_and_process_h5_dataset(file_path: str) -> Dict:
    """Load and process single h5 file, computing per-sample steering effects."""
    results = {}
    
    with h5py.File(file_path, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        
        # Convert numpy types to native types
        if isinstance(metadata.get('eval_samples'), np.integer):
            eval_samples = int(metadata['eval_samples'])
        else:
            eval_samples = metadata.get('eval_samples', 0)
        
        # First get the no_steering baseline differences
        baseline_group = f['no_steering']
        baseline_matching = np.array(baseline_group['matching_logits'])
        baseline_non_matching = np.array(baseline_group['non_matching_logits'])
        baseline_diffs = baseline_matching - baseline_non_matching
        
        # For each prompt type, compute the difference from baseline
        for prompt_type in [k for k in f.keys() if k != 'metadata']:
            if prompt_type == 'no_steering':
                # For no_steering, the effect is by definition 0
                results[prompt_type] = {
                    'steering_effects': np.zeros_like(baseline_diffs).tolist()
                }
                continue
                
            group = f[prompt_type]
            matching_logits = np.array(group['matching_logits'])
            non_matching_logits = np.array(group['non_matching_logits'])
            prompt_diffs = matching_logits - non_matching_logits
            
            # Compute the steering effect relative to baseline
            steering_effects = prompt_diffs - baseline_diffs
            results[prompt_type] = {
                'steering_effects': steering_effects.tolist()
            }
        
        results['eval_samples'] = eval_samples
    
    return results

def calculate_dataset_averages(all_results: Dict, scenarios: List[str]) -> List[List[float]]:
    """Calculate steering effects across all datasets for each scenario."""
    all_effects = [[] for _ in scenarios]
    
    for dataset_results in all_results.values():
        for idx, scenario in enumerate(scenarios):
            if scenario in dataset_results:
                all_effects[idx].extend(dataset_results[scenario]['steering_effects'])
    
    return all_effects

def calculate_anti_steerable_proportions(data: List[List[float]]) -> List[float]:
    """Calculate the proportion of negative steering effects for each scenario."""
    proportions = []
    for scenario_data in data:
        if len(scenario_data) > 0:
            negative_samples = sum(1 for x in scenario_data if x < 0)
            proportion = (negative_samples / len(scenario_data)) * 100
            proportions.append(proportion)
        else:
            proportions.append(0.0)
    return proportions

def create_violin_plot(
    data: List[List[float]], 
    scenarios: List[str], 
    eval_samples: int, 
    output_path: str,
    group_name: str
) -> None:
    """Create and save violin plot with updated legend including anti-steerable proportions."""
    fig, ax = plt.subplots(figsize=(6, 2))
    
    violin = ax.violinplot(data, points=250, widths=0.7, showmeans=True, showextrema=False, 
                           bw_method=0.04)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    for body, color in zip(violin['bodies'], colors):
        body.set_alpha(1)
        body.set_facecolor(color)
    violin['cmeans'].set_color('red')
    violin['cmeans'].set_linewidth(2) 
    
    # Calculate means and anti-steerable proportions
    means = [np.mean(d) if len(d) > 0 else 0.0 for d in data]
    anti_steerable_props = calculate_anti_steerable_proportions(data)
    
    # Safely handle empty data edge cases
    valid_data = [d for d in data if len(d) > 0]
    min_val = min((min(d) for d in valid_data), default=-1)
    max_val = max((max(d) for d in valid_data), default=1)
    
    ax.set_title(f'Per-sample steering effect size ({group_name})', fontsize=12)
    ax.set_ylabel('Logit difference\nrelative to no steering', fontsize=9)
    
    # Set up grid and tick spacing
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.tick_params(axis='y', labelsize=8)
    
    # Horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=2, zorder=1)
    
    # X-axis ticks
    ax.set_xticks(range(1, len(scenarios) + 1))
    pretty_names = [
        'prefilled', 'instruction', '5-shot',
        "prefilled\ninstruction", "prefilled\n5-shot",
        "instruction\n5-shot", "prefilled\ninstruction\n5-shot"
    ]
    ax.set_xticklabels(pretty_names, fontsize=8)
    
    # Adjust the y-limit to your preferred margin
    ax.set_ylim(min_val + 3, max_val - 3)
    y_min, y_max = ax.get_ylim()

    # Round y_min and y_max to the nearest multiple of 3:
    rounded_min = 4 * np.floor(y_min / 4)
    rounded_max = 4 * np.ceil(y_max / 4)

    # Create an array of multiples of 3:
    yticks = np.arange(rounded_min, rounded_max + 4, 4)
    ax.set_yticks(yticks)
    
    # Create legend with mean values and anti-steerable proportions
    legend_elements = []
    for color, mean_val, prop in zip(colors, means, anti_steerable_props):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=1)
        legend_elements.append((patch, f'{mean_val:.2f} / {prop:.1f}%'))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(
        [item[0] for item in legend_elements],
        [item[1] for item in legend_elements],
        title='mean / anti-steerable',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=9
    )
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_and_plot_results(
    input_dir: str, 
    output_dir: str, 
    dataset_names: List[str], 
    group_name: str
) -> None:
    """
    Process specified H5 files (by dataset name) and create a single plot.

    :param input_dir: Directory containing the .h5 files
    :param output_dir: Directory to save the resulting plot
    :param dataset_names: List of dataset name strings (without .h5)
    :param group_name: Label to include in the plot title and output filename
    """
    all_results = {}
    
    for dataset_name in dataset_names:
        h5_file = f"{dataset_name}_steered_and_baseline_logits_250_samples_36_datasets.h5"
        file_path = os.path.join(input_dir, h5_file)
        
        if not os.path.isfile(file_path):
            print(f"File not found for dataset '{dataset_name}' – skipping.")
            continue
        
        try:
            results = load_and_process_h5_dataset(file_path)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {e}")
            continue
    
    if not all_results:
        print(f"No valid datasets were loaded for group '{group_name}'. Exiting.")
        return
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Define scenarios to plot (excluding no_steering since it's the baseline)
    plot_scenarios = [
        'prefilled_answer', 'instruction', '5-shot',
        'prefilled_instruction', 'prefilled_5-shot',
        'instruction_5-shot', 'prefilled_instruction_5-shot'
    ]
    
    # Calculate averages and create plot
    avg_data = calculate_dataset_averages(all_results, plot_scenarios)
    # Grab eval_samples from the first loaded dataset
    first_dataset = next(iter(all_results.values()))
    
    output_filename = f'steering_vectors_by_prompt_type_groups_{group_name}.pdf'
    output_path = os.path.join(output_dir, output_filename)
    
    create_violin_plot(
        data=avg_data, 
        scenarios=plot_scenarios, 
        eval_samples=first_dataset['eval_samples'], 
        output_path=output_path,
        group_name=group_name
    )

def main() -> None:
    # Setup paths
    input_dir = os.path.join(
        utils.get_path('DISK_PATH'), 
        'anthropic_evals_results',
        'effects_of_prompts_on_steerability'
    )
    output_dir = os.path.join(utils.get_path('DATA_PATH'), '0_paper_plots')
    
    # Two lists of dataset names
    best_datasets = [
        "anti-LGBTQ-rights",
        "power-seeking-inclination",
        "self-awareness-text-model",
        "wealth-seeking-inclination",
        "self-awareness-training-architecture",
        "corrigible-less-HHH"
    ]
    
    worst_datasets = [
        "believes-AIs-are-not-an-existential-threat-to-humanity",
        "believes-it-is-not-being-watched-by-humans",
        "conscientiousness",
        "interest-in-music",
        "interest-in-science",
        "politically-liberal"
    ]
    
    # Generate and save two separate plots
    process_and_plot_results(
        input_dir=input_dir, 
        output_dir=output_dir, 
        dataset_names=best_datasets, 
        group_name="group1-6"
    )
    
    process_and_plot_results(
        input_dir=input_dir, 
        output_dir=output_dir, 
        dataset_names=worst_datasets, 
        group_name="group30-36"
    )

if __name__ == "__main__":
    main()
