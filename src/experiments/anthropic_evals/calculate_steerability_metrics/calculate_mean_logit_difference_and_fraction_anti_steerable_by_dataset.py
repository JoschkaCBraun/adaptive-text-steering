"""
calculate_mean_logit_difference_and_fraction_anti_steerable_by_dataset.py

Analyzes h5 files and creates visualization of steering vector effects.
Computes per-sample steering effect relative to no_steering baseline.
Additionally, computes for each dataset (by prompt type and overall):
  - average logit difference (steering effect)
  - fraction of anti-steerable samples (i.e. negative steering effect)
The results are stored in a JSON file under the DATA PATH.
"""

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import src.utils as utils

def load_and_process_h5_dataset(file_path: str) -> Dict:
    """Load and process a single h5 file, computing per-sample steering effects."""
    results = {}
    
    with h5py.File(file_path, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        
        # Convert numpy types to native types if needed
        eval_samples = int(metadata.get('eval_samples', 0))
        
        # Get the no_steering baseline differences
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

def create_violin_plot(data: List[List[float]], scenarios: List[str], 
                      eval_samples: int, output_path: str) -> None:
    """Create and save a violin plot with updated legend including anti-steerable proportions."""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    violin = ax.violinplot(data, points=250, widths=0.7, showmeans=True, 
                             showextrema=False, bw_method=0.05)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    for body, color in zip(violin['bodies'], colors):
        body.set_alpha(0.7)
        body.set_facecolor(color)
    violin['cmeans'].set_color('red')
    violin['cmeans'].set_linewidth(2)
    
    # Calculate means and anti-steerable proportions
    means = [np.mean(d) for d in data]
    anti_steerable_props = calculate_anti_steerable_proportions(data)
    
    min_val = min(min(d) for d in data if len(d) > 0)
    max_val = max(max(d) for d in data if len(d) > 0)
    
    ax.set_title('Per-Sample Steering Impact of different Prompt Types\n'
                 'Averaged across 36 datasets with 250 eval samples each',
                 fontsize=14, pad=20)
    ax.set_ylabel('Per-Sample Steering Effect', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.axhline(y=0, color='black', linewidth=2, zorder=1)
    
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=8)
    
    ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    ax.set_ylim(min_val + 2, max_val - 4)

    # Create legend with mean values and anti-steerable proportions
    legend_elements = []
    for color, mean, prop in zip(colors, means, anti_steerable_props):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7)
        legend_elements.append((patch, f'{mean:.2f} / {prop:.1f}%'))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(
        [item[0] for item in legend_elements],
        [item[1] for item in legend_elements],
        title='mean / % anti-steerable',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        title_fontsize=10
    )
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def compute_metrics_per_dataset(all_results: Dict, prompt_types: List[str]) -> Dict:
    """
    For each dataset compute:
      - For each prompt type (excluding no_steering): the average steering effect and
        the percentage of samples with negative steering effect.
      - An overall average computed across all steering prompt types.
    """
    dataset_metrics = {}
    
    for dataset_name, dataset_results in all_results.items():
        metrics = {}
        combined_effects = []
        for prompt in prompt_types:
            if prompt in dataset_results:
                effects = dataset_results[prompt]['steering_effects']
                effects_array = np.array(effects)
                mean_effect = float(np.mean(effects_array))
                anti_percentage = float((np.sum(effects_array < 0) / len(effects_array)) * 100)
                metrics[prompt] = {
                    'mean_logit_diff': mean_effect,
                    'anti_steerable_percentage': anti_percentage
                }
                # Combine effects for overall average (concatenate all sample values)
                combined_effects.extend(effects)
        # Compute overall average across prompt types if there is data
        if combined_effects:
            combined_array = np.array(combined_effects)
            overall_mean = float(np.mean(combined_array))
            overall_anti = float((np.sum(combined_array < 0) / len(combined_array)) * 100)
        else:
            overall_mean = 0.0
            overall_anti = 0.0
        metrics['average'] = {
            'mean_logit_diff': overall_mean,
            'anti_steerable_percentage': overall_anti
        }
        dataset_metrics[dataset_name] = metrics
    return dataset_metrics

def process_and_plot_results(input_dir: str, plot_output_dir: str, num_datasets: int) -> None:
    """Process h5 files, create plot, and compute/store per-dataset metrics."""
    all_results = {}
    
    # Get all h5 files from the input directory
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    h5_files = h5_files[:num_datasets]
    
    for h5_file in h5_files:
        dataset_name = h5_file.split('_steered_and_baseline_logits')[0]
        file_path = os.path.join(input_dir, h5_file)
        
        try:
            results = load_and_process_h5_dataset(file_path)
            # Only process datasets that contain a key for a steering prompt type
            if 'prefilled_answer' in results:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_results:
        print("No valid datasets processed.")
        return
    
    # Create output directory for plots if needed
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # Define the steering prompt types to plot (excluding no_steering)
    plot_scenarios = ['prefilled_answer', 'instruction', '5-shot',
                      'prefilled_instruction', 'prefilled_5-shot', 'instruction_5-shot',
                      'prefilled_instruction_5-shot']
    
    # Calculate averages for plotting
    avg_data = calculate_dataset_averages(all_results, plot_scenarios)
    first_dataset = next(iter(all_results.values()))
    
    plot_output_path = os.path.join(plot_output_dir, 
                                    'steering_vectors_from_all_prompts_work_equally_well_logits.pdf')
    create_violin_plot(avg_data, plot_scenarios, first_dataset['eval_samples'], plot_output_path)
    
    # Compute per-dataset metrics
    dataset_metrics = compute_metrics_per_dataset(all_results, plot_scenarios)
    
    # Save metrics JSON to the DATA PATH
    # Build the metrics directory from the DATA_PATH
    data_path = utils.get_path("DATA_PATH")
    metrics_dir = os.path.join(data_path, 'anthropic_evals_results', 'steerability_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_json_path = os.path.join(metrics_dir, 'steerability_metrics.json')
    
    with open(metrics_json_path, 'w') as f:
        json.dump(dataset_metrics, f, indent=4)
    
    print(f"Steerability metrics saved to {metrics_json_path}")

def main() -> None:
    # Setup paths using the get_path utility.
    disk_path = utils.get_path("DISK_PATH")
    data_path = utils.get_path("DATA_PATH")
    
    # Input directory for h5 files is based on DISK_PATH.
    input_dir = os.path.join(disk_path, 'anthropic_evals_results', 'effects_of_prompts_on_steerability')
    # Plot output directory is based on DATA_PATH.
    plot_output_dir = os.path.join(data_path, '0_paper_plots')
    
    process_and_plot_results(input_dir, plot_output_dir, num_datasets=36)

if __name__ == "__main__":
    main()
