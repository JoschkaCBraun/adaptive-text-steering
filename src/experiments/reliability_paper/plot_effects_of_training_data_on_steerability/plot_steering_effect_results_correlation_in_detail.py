import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

def get_scenario_display_name(scenario: str) -> str:
    """Convert internal scenario names to display names."""
    if scenario == 'baseline':
        return 'no steering'
    elif scenario == 'matching_prefilled_non_matching_prefilled':
        return 'original approach'
    else:
        return scenario.replace('_', '\n')

def collect_all_probs(all_results: Dict) -> Dict[str, List[float]]:
    """Collect probabilities for all scenarios across all datasets."""
    # Get list of scenarios from the first dataset
    first_dataset = next(iter(all_results.values()))
    scenarios = ['baseline'] + [f"{pos}_{neg}" for pos, neg in first_dataset['scenarios']]
    
    # Initialize dictionary to store all probabilities for each scenario
    all_probs = {scenario: [] for scenario in scenarios}
    
    # Collect probabilities from all datasets
    for dataset_results in all_results.values():
        for scenario in scenarios:
            all_probs[scenario].extend(dataset_results[scenario]['probs'])
    
    return all_probs

def create_correlation_matrix(all_results: Dict) -> None:
    """Create and plot correlation matrix between different scenarios."""
    # Collect all probabilities
    scenario_probs = collect_all_probs(all_results)
    
    # Convert to numpy arrays and create correlation matrix
    scenarios = list(scenario_probs.keys())
    prob_arrays = [scenario_probs[scenario] for scenario in scenarios]
    corr_matrix = np.corrcoef(prob_arrays)
    
    # Create better display names for the scenarios
    display_names = [get_scenario_display_name(scenario) for scenario in scenarios]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,  # Show correlation values
        fmt='.2f',   # Format to 2 decimal places
        cmap='RdBu_r',  # Red-Blue diverging colormap
        vmin=-1, vmax=1,  # Fix scale from -1 to 1
        center=0,    # Center colormap at 0
        square=True, # Make cells square
        xticklabels=display_names,
        yticklabels=display_names
    )
    
    plt.title('Correlation Matrix of Probabilities Across All Scenarios\nand All Datasets', 
              pad=20, fontsize=14)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf()

def plot_correlation_matrix(results_file: str) -> None:
    """Main function to create and save correlation matrix plot."""
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Create correlation matrix
    fig = create_correlation_matrix(all_results)
    
    # Save plot
    output_dir = os.path.dirname(results_file)
    base_name = os.path.splitext(os.path.basename(results_file))[0]
    plot_path = os.path.join(output_dir, f'{base_name}_correlation_matrix.pdf')
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    import sys
    
    default_file_path = "data/steering_analysis_results/steering_analysis_results_200_samples_20250119_214428.json"
    
    # Use command line argument if provided, otherwise use default
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_file_path
    
    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    plot_correlation_matrix(results_file)