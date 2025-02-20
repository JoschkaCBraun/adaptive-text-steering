"""
plot_sample_wise_logit_difference_for_steered_outputs.py
Analyzes h5 files and creates visualization of steering vector effects.
Computes per-sample steering effect relative to no_steering baseline.
Generates a subplot per dataset and stores all subplots in one PDF.
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

def create_violin_subplot(ax, data: List[List[float]], scenarios: List[str], 
                            eval_samples: int, dataset_name: str) -> None:
    """Create a violin plot on the given axis for a single dataset."""
    violin = ax.violinplot(data, points=250, widths=0.7, showmeans=True, 
                             showextrema=False, bw_method=0.2)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    for body, color in zip(violin['bodies'], colors):
        body.set_alpha(1)
        body.set_facecolor(color)
    violin['cmeans'].set_color('red')
    violin['cmeans'].set_linewidth(2) 
    
    # Calculate means and anti-steerable proportions
    means = [np.mean(d) for d in data]
    anti_steerable_props = calculate_anti_steerable_proportions(data)
    
    # Determine min and max values for y-axis limits
    non_empty_data = [d for d in data if len(d) > 0]
    if non_empty_data:
        min_val = min(min(d) for d in non_empty_data)
        max_val = max(max(d) for d in non_empty_data)
    else:
        min_val, max_val = -1, 1
    
    ax.set_title(f'Steering Effect Plot for {dataset_name}', fontsize=12)
    ax.set_ylabel('Logit difference\nrelative to no steering', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.axhline(y=0, color='black', linewidth=2, zorder=1)
    
    ax.set_xticks(range(1, len(scenarios) + 1))
    pretty_names = ['prefilled', 'instruction', '5-shot', "prefilled\ninstruction",
                    "prefilled\n5-shot", "instruction\n5-shot", "prefilled\ninstruction\n5-shot"]
    ax.set_xticklabels(pretty_names, fontsize=8)
    ax.set_ylim(min_val, max_val)
    
    # Create legend with mean values and anti-steerable proportions
    legend_elements = []
    for color, mean, prop in zip(colors, means, anti_steerable_props):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=1)
        legend_elements.append((patch, f'{mean:.2f} / {prop:.1f}%'))
    
    # Adjust legend placement if necessary
    ax.legend(
        [item[0] for item in legend_elements],
        [item[1] for item in legend_elements],
        title='mean / anti-steerable',
        loc='upper right',
        fontsize=9,
        title_fontsize=9
    )

def process_and_plot_results(input_dir: str, output_dir: str, num_datasets: int) -> None:
    """Process h5 files and create subplots for each dataset in one plot."""
    # Define scenarios to plot (excluding no_steering since it's the baseline)
    plot_scenarios = ['prefilled_answer', 'instruction', '5-shot',
                      'prefilled_instruction', 'prefilled_5-shot', 'instruction_5-shot',
                      'prefilled_instruction_5-shot']
    
    # Get all h5 files in the input directory
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    h5_files = h5_files[:num_datasets]
    
    # List to store tuples: (dataset_name, eval_samples, dataset_data)
    datasets_to_plot = []
    
    for h5_file in h5_files:
        # Extract dataset name from file name (everything before "_steered_and_baseline_logits")
        if "_steered_and_baseline_logits" not in h5_file:
            print(f"File {h5_file} does not match naming pattern. Skipping.")
            continue
        dataset_name = h5_file.split("_steered_and_baseline_logits")[0]
        file_path = os.path.join(input_dir, h5_file)
        
        try:
            dataset_results = load_and_process_h5_dataset(file_path)
        except Exception as e:
            print(f"Error processing file {h5_file}: {e}")
            continue
        
        # Check for required key
        if 'prefilled_answer' not in dataset_results:
            print(f"Error: 'prefilled_answer' not found in {h5_file}. Skipping.")
            continue
        
        # Extract data for each scenario
        dataset_data = []
        missing_key = False
        for scenario in plot_scenarios:
            if scenario not in dataset_results:
                print(f"Error: '{scenario}' not found in {h5_file}. Skipping this dataset.")
                missing_key = True
                break
            dataset_data.append(dataset_results[scenario]['steering_effects'])
        
        if missing_key:
            continue
        
        eval_samples = dataset_results.get('eval_samples', 0)
        datasets_to_plot.append((dataset_name, eval_samples, dataset_data))
    
    if not datasets_to_plot:
        print("No valid datasets to plot.")
        return
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with one subplot per dataset.
    num_plots = len(datasets_to_plot)
    fig_height = 3 * num_plots  # adjust as needed for readability
    fig, axes = plt.subplots(num_plots, 1, figsize=(7, fig_height), squeeze=False)
    
    for idx, (dataset_name, eval_samples, dataset_data) in enumerate(datasets_to_plot):
        ax = axes[idx, 0]
        create_violin_subplot(ax, dataset_data, plot_scenarios, eval_samples, dataset_name)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'logit_differences_all_datasets.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")

def main() -> None:
    # Setup paths
    input_dir = os.path.join(utils.get_path('DISK_PATH'), 'anthropic_evals_results',
                             'effects_of_prompts_on_steerability')
    output_dir = os.path.join(utils.get_path('DATA_PATH'), '0_paper_plots')
    
    process_and_plot_results(input_dir, output_dir, num_datasets=36)

if __name__ == "__main__":
    main()
