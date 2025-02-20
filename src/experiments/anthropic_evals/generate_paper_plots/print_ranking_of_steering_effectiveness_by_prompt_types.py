"""
visualize_prompt_type_rankings.py
Analyzes h5 files to compute ranking counts for each prompt type, and then visualizes
the counts using a stacked bar chart. The colors range from green (Rank 1) to red (Rank 7).
The figure is stored in the output directory: DATA_PATH/0_paper_plots.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pprint import pprint

def load_and_process_h5_dataset(file_path: str) -> Dict:
    """Load and process a single h5 file, computing per-sample steering effects."""
    results = {}
    
    with h5py.File(file_path, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        
        # Convert numpy types to native types
        if isinstance(metadata.get('eval_samples'), np.integer):
            eval_samples = int(metadata['eval_samples'])
        else:
            eval_samples = metadata.get('eval_samples', 0)
        
        # Get the no_steering baseline differences
        baseline_group = f['no_steering']
        baseline_matching = np.array(baseline_group['matching_logits'])
        baseline_non_matching = np.array(baseline_group['non_matching_logits'])
        baseline_diffs = baseline_matching - baseline_non_matching
        
        # Process each prompt type
        for prompt_type in [k for k in f.keys() if k != 'metadata']:
            if prompt_type == 'no_steering':
                # The effect is 0 for the baseline
                results[prompt_type] = {'steering_effects': np.zeros_like(baseline_diffs).tolist()}
                continue
            
            group = f[prompt_type]
            matching_logits = np.array(group['matching_logits'])
            non_matching_logits = np.array(group['non_matching_logits'])
            prompt_diffs = matching_logits - non_matching_logits
            
            # Compute the steering effect relative to baseline
            steering_effects = prompt_diffs - baseline_diffs
            results[prompt_type] = {'steering_effects': steering_effects.tolist()}
        
        results['eval_samples'] = eval_samples
    
    return results

def calculate_ranking_counts(all_results: Dict, scenarios: List[str]) -> Dict[str, Dict[int, int]]:
    """
    For each dataset, calculate the average steering effect per scenario,
    rank the prompt types (1 = highest average, 2 = second highest, etc.),
    and tally the counts across datasets.
    Returns a dictionary where keys are prompt types and values are dictionaries
    mapping rank (1 ... len(scenarios)) to count.
    """
    # Initialize ranking_counts: for each scenario, count ranks from 1 to len(scenarios)
    ranking_counts = {scenario: {rank: 0 for rank in range(1, len(scenarios) + 1)} for scenario in scenarios}
    
    for dataset_results in all_results.values():
        avg_dict = {}
        for scenario in scenarios:
            if scenario in dataset_results:
                effects = dataset_results[scenario]['steering_effects']
                avg_effect = np.mean(effects) if effects else 0.0
                avg_dict[scenario] = avg_effect
            else:
                avg_dict[scenario] = -np.inf  # If missing, treat as worst
                
        # Sort prompt types by average effect in descending order (higher is better)
        sorted_prompts = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
        # Update ranking counts: rank 1 for highest, 2 for second, etc.
        for rank, (scenario, _) in enumerate(sorted_prompts, start=1):
            ranking_counts[scenario][rank] += 1
            
    return ranking_counts

def process_results(input_dir: str, num_datasets: int) -> (Dict[str, Dict[int, int]], List[str]):
    """Process h5 files and calculate ranking counts."""
    all_results = {}
    
    # Get all h5 files in the input directory
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    h5_files = h5_files[:num_datasets]
    
    # Load all datasets
    for h5_file in h5_files:
        dataset_name = h5_file.split('_steered_and_baseline_logits')[0]
        file_path = os.path.join(input_dir, h5_file)
        try:
            results = load_and_process_h5_dataset(file_path)
            # Only include dataset if it contains the key 'prefilled_answer'
            if 'prefilled_answer' in results:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_results:
        print("No valid results found.")
        return {}, []
    
    # Define scenarios to consider (excluding no_steering since it's the baseline)
    scenarios = ['prefilled_answer', 'instruction', '5-shot',
                 'prefilled_instruction', 'prefilled_5-shot', 'instruction_5-shot',
                 'prefilled_instruction_5-shot']
    
    ranking_counts = calculate_ranking_counts(all_results, scenarios)
    print("Ranking counts (prompt type: {rank: count, ...}):")
    pprint(ranking_counts)
    
    return ranking_counts, scenarios

def plot_ranking_counts(ranking_counts: Dict[str, Dict[int, int]], scenarios: List[str], output_dir: str) -> None:
    """
    Visualize the ranking counts as a stacked bar chart.
    
    Each prompt type (x-axis) shows the number of times it ranked 1st, 2nd, ...,
    represented as segments in a stacked bar. The colors range from green (Rank 1) to red (Rank 7).
    """
    # Determine the ranking positions (assumes ranks 1 ... len(scenarios))
    ranks = sorted(next(iter(ranking_counts.values())).keys())  # e.g., [1, 2, ..., 7]
    
    # Use the given scenarios order for the x-axis
    prompt_types = scenarios  
    # Prepare data: a matrix where each row corresponds to a prompt type and each column to a rank.
    data = np.array([[ranking_counts[pt][rank] for rank in ranks] for pt in prompt_types])
    
    # Define color mapping from rank 1 (green) to rank 7 (red) using a discrete colormap.
    # Using "RdYlGn_r" with 7 discrete colors: index 0 -> green, index 6 -> red.
    cmap = plt.cm.get_cmap("RdYlGn_r", len(ranks))
    color_list = [cmap(i) for i in range(len(ranks))]
    
    # Create a smaller stacked bar chart.
    fig, ax = plt.subplots(figsize=(5, 4))
    bottom = np.zeros(len(prompt_types))
    
    for idx, rank in enumerate(ranks):
        counts = data[:, idx]
        # Label the bars with just the rank number (as a string)
        ax.bar(prompt_types, counts, bottom=bottom, label=str(rank), color=color_list[idx])
        bottom += counts
    
    # Remove the prompt_type x-axis label (x-axis ticks remain, but no label is set)
    # ax.set_xlabel('Prompt Type')  <-- This line is removed.
    ax.set_ylabel('Count')
    ax.set_title('Ranking Counts for Prompt Types')
    ax.legend(title='Rank', bbox_to_anchor=(1.01, 1.04), loc='upper left',
              fontsize=12, title_fontsize=12, markerscale=1, handleheight=2, handlelength=1)
    pretty_names = ['prefilled', '\n\n\ninstruction', '5-shot', "\n\n\nprefilled\ninstruction",
            "prefilled\n5-shot", "\n\n\ninstruction\n5-shot", "prefilled\ninstruction\n5-shot"]
    plt.xticks(ticks=range(len(pretty_names)), labels=pretty_names, fontsize=9, rotation=0)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'prompt_type_ranking_counts.pdf')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure saved to: {output_file}")

def main() -> None:
    # Setup the input directory (adjust DISK_PATH as needed)
    input_dir = os.path.join(os.environ.get('DISK_PATH', ''), 'anthropic_evals_results',
                             'effects_of_prompts_on_steerability')
    num_datasets = 36  # Adjust if needed
    
    # Setup the output directory (DATA_PATH/0_paper_plots)
    output_dir = os.path.join(os.environ.get('DATA_PATH', ''), '0_paper_plots')
    
    ranking_counts, scenarios = process_results(input_dir, num_datasets)
    if ranking_counts and scenarios:
        plot_ranking_counts(ranking_counts, scenarios, output_dir)

if __name__ == "__main__":
    main()
