"""
select_subset_of_datasets_and_plot.py
Analyzes steering data and plots only datasets where prefilled_answer shows improvement above threshold.
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import src.utils as utils

def calculate_mean_probs(probs_list: List[float]) -> float:
    """Calculate mean probability, handling empty lists."""
    return np.mean(probs_list) if probs_list else 0.0

def get_selected_datasets(data: Dict, selection_threshold: float) -> Tuple[Dict, Dict]:
    """
    Select datasets where prefilled_answer mean is higher than no_steering mean + threshold.
    Returns both selected and rejected datasets with their differences.
    """
    selected_datasets = {}
    rejected_datasets = {}
    
    for dataset_name, dataset_info in data.items():
        no_steering_mean = calculate_mean_probs(dataset_info['no_steering']['probs'])
        prefilled_mean = calculate_mean_probs(dataset_info['prefilled_answer']['probs'])
        difference = prefilled_mean - no_steering_mean
        
        if difference > selection_threshold:
            selected_datasets[dataset_name] = dataset_info
        else:
            rejected_datasets[dataset_name] = {
                'difference': difference,
                'no_steering_mean': no_steering_mean,
                'prefilled_mean': prefilled_mean
            }
    
    return selected_datasets, rejected_datasets

def main() -> None:
    # Parameters
    selection_threshold = 0.1
    
    # Load data
    input_path = 'data/anthropic_evals_results/impact_of_steering_on_probs/steering_analysis_results_100_samples_20250130_123358.json'
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Select datasets
    selected_datasets, rejected_datasets = get_selected_datasets(data, selection_threshold)
    
    # Print information about rejected datasets
    print("\nRejected Datasets (prefilled_answer improvement below threshold):")
    print("-" * 80)
    print(f"{'Dataset Name':<40} {'Difference':>10} {'No Steering':>12} {'Prefilled':>12}")
    print("-" * 80)
    for dataset_name, info in rejected_datasets.items():
        print(f"{dataset_name:<40} {info['difference']:10.4f} {info['no_steering_mean']:12.4f} {info['prefilled_mean']:12.4f}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(input_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save selected datasets to a new JSON file
    selected_datasets_path = os.path.join(
        output_dir,
        f'selected_datasets_threshold_{selection_threshold}_{timestamp}.json'
    )
    
    with open(selected_datasets_path, 'w') as f:
        json.dump(selected_datasets, f, indent=2)
    
    # Generate plots using utils.plot_all_results
    plot_scenarios = ['no_steering', 'prefilled_answer', 'instruction', '5-shot',
                     'prefilled_instruction', 'prefilled_5-shot', 'instruction_5-shot',
                     'prefilled_instruction_5-shot']
    
    print(f"\nGenerating plots for {len(selected_datasets)} selected datasets...")
    utils.plot_all_results(selected_datasets_path, plot_scenarios)
    print(f"Results saved with selected datasets (threshold = {selection_threshold})")
    print(f"\nSelected {len(selected_datasets)} datasets out of {len(data)} total datasets")

if __name__ == "__main__":
    main()