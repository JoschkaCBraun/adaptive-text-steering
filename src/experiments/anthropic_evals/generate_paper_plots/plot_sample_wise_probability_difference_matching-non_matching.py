"""
analyze_stored_effects_of_prompts.py
Analyzes stored h5 files and creates violin plots of probabilities and probability differences.
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import src.utils as utils

def load_h5_dataset(file_path: str) -> Dict:
    """Load probabilities and their differences from h5 file."""
    results = {}
    
    with h5py.File(file_path, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        metadata = convert_to_native_types(metadata)
        
        for prompt_type in ['no_steering'] + [k for k in f.keys() if k != 'metadata']:
            group = f[prompt_type]
            
            # Directly access stored probabilities
            matching_probs = np.array(group['matching_probs'])
            non_matching_probs = np.array(group['non_matching_probs'])
            
            # Calculate differences relative to no_steering baseline
            if prompt_type == 'no_steering':
                prob_differences = np.zeros_like(matching_probs)  # baseline case
            else:
                # Get baseline probabilities from no_steering case
                baseline_matching = np.array(f['no_steering']['matching_probs'])
                baseline_non_matching = np.array(f['no_steering']['non_matching_probs'])
                baseline_diff = baseline_matching - baseline_non_matching
                
                # Calculate current differences and subtract baseline
                current_diff = matching_probs - non_matching_probs
                prob_differences = current_diff - baseline_diff
            
            results[prompt_type] = {
                'matching_probs': convert_to_native_types(matching_probs),
                'non_matching_probs': convert_to_native_types(non_matching_probs),
                'prob_differences': convert_to_native_types(prob_differences)
            }
        
        results.update({
            'dataset_type': metadata.get('dataset_type', 'unknown'),
            'train_samples': metadata['train_samples'],
            'eval_samples': metadata['eval_samples'],
            'model_layer': metadata['model_layer'],
            'steering_multiplier': metadata['steering_multiplier']
        })
        
    return results

def convert_to_native_types(obj) -> object:
    """Convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    return obj

def calculate_dataset_averages(all_results: Dict, scenarios: List[str], 
                             metric: str = 'matching_probs') -> List[List[float]]:
    """Calculate average effects across all datasets for each scenario."""
    all_values = [[] for _ in scenarios]
    
    for dataset_results in all_results.values():
        for idx, scenario in enumerate(scenarios):
            if scenario in dataset_results:
                all_values[idx].extend(dataset_results[scenario][metric])
    
    return all_values

def create_violin_plot(data: List[List[float]], scenarios: List[str], eval_samples: int,
                      output_path: str, plot_title: str, ylabel: str) -> None:
    """Create and save violin plot."""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    violin = ax.violinplot(data, points=100, widths=0.7, showmeans=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    for body, color in zip(violin['bodies'], colors):
        body.set_alpha(0.7)
        body.set_facecolor(color)
    violin['cmeans'].set_color('red')
    violin['cmeans'].set_linewidth(2)
    
    # Calculate means for legend
    means = [np.mean(d) for d in data]
    min_val = min(min(d) for d in data if len(d) > 0)
    max_val = max(max(d) for d in data if len(d) > 0)
    
    ax.set_title(f'{plot_title}\nBased on {eval_samples} eval samples per dataset',
                 fontsize=14, pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.axhline(y=0, color='black', linewidth=2, zorder=1)
    
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=8)
    
    padding = (max_val - min_val) * 0.05
    ax.set_ylim(min_val - padding, max_val + padding)
    
    # Create legend with mean values
    legend_elements = []
    for color, mean, scenario in zip(colors, means, scenarios):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7)
        legend_elements.append((patch, f'{mean*100:.1f}%'))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        [item[0] for item in legend_elements],
        [item[1] for item in legend_elements],
        title='Mean Difference',
        loc='center left',
        bbox_to_anchor=(1.1, 0.5),
        fontsize=10,
        title_fontsize=12
    )
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_and_plot_results(input_dir: str, num_datasets: int) -> None:
    """Process h5 files and create violin plots."""
    all_results = {}
    
    # Get all h5 files
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    h5_files = h5_files[:num_datasets]
    
    print(f"\nFound {len(h5_files)} h5 files:")
    for h5_file in h5_files:
        print(f"- {h5_file}")
    
    for h5_file in h5_files:
        dataset_name = h5_file.split('_steered_and_baseline_logits')[0]
        file_path = os.path.join(input_dir, h5_file)
        
        try:
            results = load_h5_dataset(file_path)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing {h5_file}: {str(e)}")
            continue
    
    if not all_results:
        print("No results to process!")
        return

    # Define scenarios to plot
    scenarios = ['no_steering', 'prefilled_answer', 'instruction', '5-shot',
                'prefilled_instruction', 'prefilled_5-shot', 'instruction_5-shot',
                'prefilled_instruction_5-shot']

    # Create output directory if it doesn't exist
    output_dir = os.path.join(utils.get_path("DATA_PATH"), '0_paper_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    first_dataset = next(iter(all_results.values()))
    
    # Create probability distribution plot
    avg_probs = calculate_dataset_averages(all_results, scenarios, 'matching_probs')
    create_violin_plot(
        avg_probs, 
        scenarios,
        first_dataset['eval_samples'],
        os.path.join(output_dir, 'matching_probabilities.pdf'),
        'Average Matching Probabilities Across All Datasets',
        'Probability'
    )
    
    # Create probability difference plot
    avg_diffs = calculate_dataset_averages(all_results, scenarios, 'prob_differences')
    create_violin_plot(
        avg_diffs,
        scenarios,
        first_dataset['eval_samples'],
        os.path.join(output_dir, 'probability_differences.pdf'),
        'Average Probability Differences (Matching - Non-matching) Across All Datasets',
        'Probability Difference'
    )
    
    print(f"\nProcessed {len(all_results)} datasets")
    print("Plots saved to:")
    print(f"- {output_dir}/matching_probabilities.pdf")
    print(f"- {output_dir}/probability_differences.pdf")

def main() -> None:
    # Setup paths
    input_dir = os.path.join(utils.get_path('DISK_PATH'), 'anthropic_evals_results',
                            'effects_of_prompts_on_steerability')
    
    process_and_plot_results(input_dir, num_datasets=36)

if __name__ == "__main__":
    main()