"""
plot_results_as_violine_plots.py
Creates visualization of steering vector effects from JSON results file.
Simplified version that plots probabilities for given scenario keys.
"""
import os
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def calculate_dataset_averages(all_results: Dict, scenarios: List[str]) -> List[List[float]]:
    """Calculate average effects across all datasets for each scenario.
    
    Args:
        all_results: Dictionary containing all results
        scenarios: List of scenario keys to process
    Returns:
        List of probability distributions for each scenario
    """
    all_probs = [[] for _ in scenarios]
    
    for dataset_results in all_results.values():
        for idx, scenario in enumerate(scenarios):
            if scenario in dataset_results:
                all_probs[idx].extend(dataset_results[scenario]['probs'])
    
    return all_probs

def create_violin_subplot(ax: plt.Axes, data: List[List[float]], scenarios: List[str], 
                         title: str) -> None:
    """Create a violin plot subplot with given data and mean values in legend."""
    violin = ax.violinplot(data, points=100, widths=0.7, showmeans=True)
    
    # Set colors and style
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
    
    # Set title and style
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel('Probability', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add prominent zero line
    ax.axhline(y=0, color='black', linewidth=2, zorder=1)
    
    # Set x-axis labels
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=8)
    
    # Set y-axis limits
    ax.set_ylim(min(0, min_val) - 0.05, max(1, max_val) + 0.05)
    
    # Create legend with mean values
    legend_elements = []
    for color, mean, scenario in zip(colors, means, scenarios):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7)
        legend_elements.append((patch, f'{scenario}: {mean:.3f}'))
    
    # Position legend outside plot
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

def plot_all_results(results_file: str, scenarios: List[str]) -> None:
    """Create visualization for all datasets.
    
    Args:
        results_file: Path to the JSON results file
        scenarios: List of scenario keys to plot
    """
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    num_datasets = len(all_results)
    first_dataset = next(iter(all_results.values()))
    
    # Create figure with subplots (average + individual datasets)
    fig, axes = plt.subplots(num_datasets + 1, 1, 
                            figsize=(10, 4 * (num_datasets + 1)))
    
    if num_datasets == 0:
        axes = np.array([axes])
    
    # Plot average across all datasets
    avg_probs = calculate_dataset_averages(all_results, scenarios)
    create_violin_subplot(
        axes[0], 
        avg_probs, 
        scenarios,
        f'Average Impact on Probabilities Across All Datasets\nEval samples per dataset: {first_dataset["eval_samples"]}'
    )
    
    # Plot individual datasets
    for idx, (dataset_name, results) in enumerate(all_results.items(), 1):
        dataset_distributions = []
        for scenario in scenarios:
            if scenario in results:
                dataset_distributions.append(results[scenario]['probs'])
            else:
                print(f"Warning: Scenario {scenario} not found in dataset {dataset_name}")
                dataset_distributions.append([])
        
        create_violin_subplot(
            axes[idx],
            dataset_distributions,
            scenarios,
            f'Impact on Probabilities - {dataset_name}\nEval samples: {results["eval_samples"]}'
        )
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save plot
    output_dir = os.path.dirname(results_file)
    base_name = os.path.splitext(os.path.basename(results_file))[0]
    plot_path = os.path.join(output_dir, f'{base_name}_plots.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()