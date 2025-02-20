import os
import sys
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def get_scenario_display_name(scenario: str) -> str:
    """Convert internal scenario names to display names."""
    if scenario == 'baseline':
        return 'no steering'
    elif scenario == 'matching_prefilled_non_matching_prefilled':
        return 'original approach /\nbaseline:\nmatching prefilled\nnon matching prefilled'
    else:
        return scenario.replace('_', '\n')

def calculate_dataset_averages(all_results: Dict) -> Tuple[List[List[float]], List[List[float]]]:
    """Calculate average effects across all datasets for each scenario."""
    first_dataset = next(iter(all_results.values()))
    scenarios = ['baseline'] + [f"{pos}_{neg}" for pos, neg in first_dataset['scenarios']]
    
    all_logits = [[] for _ in scenarios]
    all_probs = [[] for _ in scenarios]
    
    for dataset_results in all_results.values():
        for i, scenario in enumerate(scenarios):
            all_logits[i].extend(dataset_results[scenario]['logits'])
            all_probs[i].extend(dataset_results[scenario]['probs'])
    
    return all_logits, all_probs

def calculate_group_averages(group_results: Dict) -> Tuple[List[List[float]], List[List[float]]]:
    """Calculate average effects for a group of datasets."""
    first_dataset = next(iter(group_results.values()))
    scenarios = ['baseline'] + [f"{pos}_{neg}" for pos, neg in first_dataset['scenarios']]
    
    all_logits = [[] for _ in scenarios]
    all_probs = [[] for _ in scenarios]
    
    for dataset_results in group_results.values():
        for i, scenario in enumerate(scenarios):
            all_logits[i].extend(dataset_results[scenario]['logits'])
            all_probs[i].extend(dataset_results[scenario]['probs'])
    
    return all_logits, all_probs

def create_violin_subplot(ax: plt.Axes, data: List[List[float]], scenarios: List[str], 
                         title: str, ylabel: str, is_probability: bool = False) -> None:
    """Create a violin plot subplot with given data and mean values in legend."""
    violin = ax.violinplot(data, points=100, widths=0.7, showmeans=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    for body, color in zip(violin['bodies'], colors):
        body.set_alpha(0.7)
        body.set_facecolor(color)
    violin['cmeans'].set_color('red')
    violin['cmeans'].set_linewidth(2)
    
    means = [np.mean(d) for d in data]
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels([get_scenario_display_name(s) for s in scenarios],
                       rotation=20, ha='right', fontsize=8)
    
    if is_probability:
        ax.set_ylim(0, 1)
    
    legend_elements = []
    for color, mean in zip(colors, means):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7)
        legend_elements.append((patch, f'mean: {mean:.3f}'))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        [item[0] for item in legend_elements],
        [item[1] for item in legend_elements],
        title='Average Values',
        loc='center left',
        bbox_to_anchor=(1.1, 0.5),
        fontsize=10,
        title_fontsize=12
    )

def get_dataset_groups(all_results: Dict, group_size: int) -> List[Dict]:
    """Split datasets into groups of specified size."""
    datasets = list(all_results.items())
    groups = []
    
    for i in range(0, len(datasets), group_size):
        group_dict = dict(datasets[i:i + group_size])
        groups.append(group_dict)
    
    return groups

def plot_all_results(results_file: str, plot_logits: bool = True, group_size: int = 1) -> None:
    """Create combined visualization for all datasets.
    
    Args:
        results_file (str): Path to the JSON results file
        plot_logits (bool): If True, plots both logits and probabilities. If False, plots only probabilities
        group_size (int): Number of datasets to combine in each group plot (default: 1 for individual plots)
    """
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Get dataset groups
    dataset_groups = get_dataset_groups(all_results, group_size)
    num_groups = len(dataset_groups)
    
    # Calculate total number of plots needed
    plots_per_group = 2 if plot_logits else 1
    total_rows = (num_groups + 1) * plots_per_group  # +1 for the overall average
    
    # Create figure
    fig, axes = plt.subplots(total_rows, 1, 
                            figsize=(9, 6 * (num_groups + 1)))
    
    # Convert axes to array if only one subplot
    if total_rows == 1:
        axes = np.array([axes])
    
    # Get scenario info from first dataset
    first_dataset = next(iter(all_results.values()))
    scenarios = ['baseline'] + [f"{pos}_{neg}" for pos, neg in first_dataset['scenarios']]
    
    current_ax_idx = 0
    
    # Plot overall average first
    avg_logits, avg_probs = calculate_dataset_averages(all_results)
    
    if plot_logits:
        create_violin_subplot(
            axes[current_ax_idx], 
            avg_logits, 
            scenarios,
            title=f'Average Impact on Logits Across All Datasets\nEval samples per dataset: {first_dataset["eval_samples"]}',
            ylabel='Logit Value'
        )
        current_ax_idx += 1
    
    create_violin_subplot(
        axes[current_ax_idx], 
        avg_probs, 
        scenarios,
        title=f'Average Impact on Probabilities Across All Datasets\nEval samples per dataset: {first_dataset["eval_samples"]}',
        ylabel='Probability',
        is_probability=True
    )
    current_ax_idx += 1
    
    # Plot each group
    for group_idx, group_results in enumerate(dataset_groups):
        group_names = ", ".join(group_results.keys())
        
        # Calculate averages for this group
        group_logits, group_probs = calculate_group_averages(group_results)
        
        if plot_logits:
            create_violin_subplot(
                axes[current_ax_idx],
                group_logits,
                scenarios,
                title=f'Impact on Logits - Group {group_idx + 1}: {group_names}\nEval samples: {first_dataset["eval_samples"]}',
                ylabel='Logit Value'
            )
            current_ax_idx += 1
        
        create_violin_subplot(
            axes[current_ax_idx],
            group_probs,
            scenarios,
            title=f'Impact on Probabilities - Group {group_idx + 1}: {group_names}\nEval samples: {first_dataset["eval_samples"]}',
            ylabel='Probability',
            is_probability=True
        )
        current_ax_idx += 1
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # room for legends on the right
    
    # Save combined plot
    output_dir = os.path.dirname(results_file)
    base_name = os.path.splitext(os.path.basename(results_file))[0]
    suffix = '_probs_only' if not plot_logits else '_full'
    suffix += f'_group_size_{group_size}'
    plot_path = os.path.join(output_dir, f'{base_name}{suffix}_plots.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    default_file_path = "data/steering_analysis_results/steering_analysis_results_200_samples_20250119_214428.json"

    # Use command line argument if provided, otherwise use default
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_file_path
    
    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    # Example usage with different group sizes:
    # plot_all_results(results_file, plot_logits=True, group_size=1)  # Default behavior
    # plot_all_results(results_file, plot_logits=True, group_size=3)  # Group datasets by 3
    plot_all_results(results_file, plot_logits=False, group_size=6)  # Plot probabilities only