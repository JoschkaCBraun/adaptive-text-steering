import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import src.utils as utils

def load_and_process_h5_dataset(file_path: str) -> Dict:
    """Load and process a single .h5 file, returning steering-effect data."""
    results = {}
    
    with h5py.File(file_path, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        
        # Convert numpy types to native Python types
        if isinstance(metadata.get('eval_samples'), np.integer):
            eval_samples = int(metadata['eval_samples'])
        else:
            eval_samples = metadata.get('eval_samples', 0)
        
        # Baseline group
        baseline_group = f['no_steering']
        baseline_matching = np.array(baseline_group['matching_logits'])
        baseline_non_matching = np.array(baseline_group['non_matching_logits'])
        baseline_diffs = baseline_matching - baseline_non_matching
        
        # For each prompt type, compute difference from baseline
        for prompt_type in [k for k in f.keys() if k != 'metadata']:
            if prompt_type == 'no_steering':
                results[prompt_type] = {
                    'steering_effects': np.zeros_like(baseline_diffs).tolist()
                }
                continue
                
            group = f[prompt_type]
            matching_logits = np.array(group['matching_logits'])
            non_matching_logits = np.array(group['non_matching_logits'])
            prompt_diffs = matching_logits - non_matching_logits
            
            # Steering effect relative to baseline
            steering_effects = prompt_diffs - baseline_diffs
            results[prompt_type] = {
                'steering_effects': steering_effects.tolist()
            }
        
        results['eval_samples'] = eval_samples
    
    return results

def calculate_dataset_averages(all_results: Dict, scenarios: List[str]) -> List[List[float]]:
    """Combine steering effects across all datasets for each scenario."""
    all_effects = [[] for _ in scenarios]
    
    for dataset_results in all_results.values():
        for idx, scenario in enumerate(scenarios):
            if scenario in dataset_results:
                all_effects[idx].extend(dataset_results[scenario]['steering_effects'])
    
    return all_effects

def process_datasets(
    input_dir: str,
    dataset_names: List[str],
) -> (List[List[float]], int):
    """
    Loads multiple .h5 files (by dataset name), merges results into a
    list of lists (steering_effect data, one list per scenario),
    and returns that data along with eval_samples from the first valid dataset.
    """
    all_results = {}
    
    for ds_name in dataset_names:
        h5_file = f"{ds_name}_steered_and_baseline_logits_250_samples_36_datasets.h5"
        file_path = os.path.join(input_dir, h5_file)
        
        if not os.path.isfile(file_path):
            print(f"File not found for dataset '{ds_name}' – skipping.")
            continue
        
        try:
            results = load_and_process_h5_dataset(file_path)
            all_results[ds_name] = results
        except Exception as e:
            print(f"Error loading dataset '{ds_name}': {e}")
            continue
    
    if not all_results:
        # Return empty data if no valid datasets
        return [[] for _ in range(7)], 0
    
    # Scenarios to aggregate (excluding 'no_steering' baseline)
    plot_scenarios = [
        'prefilled_answer', 'instruction', '5-shot',
        'prefilled_instruction', 'prefilled_5-shot',
        'instruction_5-shot', 'prefilled_instruction_5-shot'
    ]
    
    avg_data = calculate_dataset_averages(all_results, plot_scenarios)
    # Grab eval_samples from the first loaded dataset
    first_dataset = next(iter(all_results.values()))
    eval_samples = first_dataset['eval_samples']
    
    return avg_data, eval_samples

def calculate_anti_steerable_proportions(data: List[List[float]]) -> List[float]:
    """
    Given a list of lists, each containing steering effects,
    compute proportion of negative samples in each sub-list.
    """
    proportions = []
    for scenario_data in data:
        if len(scenario_data) > 0:
            negative_samples = sum(1 for x in scenario_data if x < 0)
            proportion = (negative_samples / len(scenario_data)) * 100
            proportions.append(proportion)
        else:
            proportions.append(0.0)
    return proportions

def create_violin_subplot(
    ax: plt.Axes,
    data: List[List[float]],
    scenarios: List[str],
    group_name: str,
    show_xlabels: bool = True  # New flag: determines if we show x-labels
):
    """
    Create a violin plot on the provided Axes, matching previous style.
    Only show x-tick labels if `show_xlabels=True`.
    """
    violin = ax.violinplot(
        data,
        points=250,
        widths=0.7,
        showmeans=True,
        showextrema=False,
        bw_method=0.04
    )
    
    # Color setup
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    for body, color in zip(violin['bodies'], colors):
        body.set_alpha(1)
        body.set_facecolor(color)
    violin['cmeans'].set_color('red')
    violin['cmeans'].set_linewidth(2) 
    
    # Means and anti-steerable
    means = [np.mean(d) if len(d) > 0 else 0.0 for d in data]
    anti_steerable_props = calculate_anti_steerable_proportions(data)
    
    # Y-limits based on actual data
    valid_data = [d for d in data if len(d) > 0]
    min_val = min((min(d) for d in valid_data), default=-1)
    max_val = max((max(d) for d in valid_data), default=1)
    
    ax.set_title(group_name, fontsize=10)
    ax.set_ylabel('Logit difference\nrelative to no steering', fontsize=8)
    
    # Grid, ticks
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.tick_params(axis='y', labelsize=8)
    ax.axhline(y=0, color='black', linewidth=2, zorder=1)
    
    # X-axis ticks with your "pretty" labeling
    pretty_names = [
        'prefilled', 'instruction', '5-shot',
        "prefilled\ninstruction", "prefilled\n5-shot",
        "instruction\n5-shot", "prefilled\ninstruction\n5-shot"
    ]
    ax.set_xticks(range(1, len(scenarios) + 1))
    
    # Conditionally show or hide the x-tick labels
    if show_xlabels:
        ax.set_xticklabels(pretty_names, fontsize=8)
    else:
        ax.set_xticklabels([])  # Hide labels
    
    # Adjust vertical range (the +3/-3 offset is from your original code)
    ax.set_ylim(min_val + 2.5, max_val - 3)
    
    # Rounded y-ticks to nearest multiple of 4
    y_min, y_max = ax.get_ylim()
    rounded_min = 4 * np.floor(y_min / 4)
    rounded_max = 4 * np.ceil(y_max / 4)
    yticks = np.arange(rounded_min, rounded_max + 4, 4)
    yticks = np.arange(-10, 15, 4)
    ax.set_yticks(yticks)
    
    # Prepare legend (for scenario colors) with mean & anti-steerable
    legend_elements = []
    for color, mean_val, prop in zip(colors, means, anti_steerable_props):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=1)
        legend_elements.append((patch, f'{mean_val:.2f} / {prop:.1f}%'))
    
    # Adjust the position a bit to fit the legend inside the Ax
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    
    # Make legend title: e.g. "mean/anti-steerable"
    ax.legend(
        [item[0] for item in legend_elements],
        [item[1] for item in legend_elements],
        title='mean/anti-steerable',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=8
    )

def main():
    input_dir = os.path.join(
        utils.get_path('DISK_PATH'), 
        'anthropic_evals_results',
        'effects_of_prompts_on_steerability'
    )
    output_dir = os.path.join(utils.get_path('DATA_PATH'), '0_paper_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Example subsets
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
    
    # For demonstration, "average" is the union of best + worst
    average_datasets = list(set(best_datasets + worst_datasets))
    
    # Gather data
    group1_data, _ = process_datasets(input_dir, best_datasets)
    avg_data, _    = process_datasets(input_dir, average_datasets)
    group2_data, _ = process_datasets(input_dir, worst_datasets)
    
    # The 7 scenarios we'll plot
    plot_scenarios = [
        'prefilled_answer', 'instruction', '5-shot',
        'prefilled_instruction', 'prefilled_5-shot',
        'instruction_5-shot', 'prefilled_instruction_5-shot'
    ]
    
    # Create a figure with 3 subplots VERTICALLY
    fig, axes = plt.subplots(3, 1, figsize=(6, 5))
    fig.suptitle("Per-sample steering effect size by prompt type", fontsize=14)
    
    # 1) Subplot for Group 1-6 (hide xlabels)
    create_violin_subplot(
        ax=axes[0],
        data=group1_data,
        scenarios=plot_scenarios,
        group_name="Most steerable datasets (group 1-6)",
        show_xlabels=False  # Hide for the first subplot
    )
    # 2) Subplot for Average (hide xlabels)
    create_violin_subplot(
        ax=axes[1],
        data=avg_data,
        scenarios=plot_scenarios,
        group_name="Average across all 36 datasets",
        show_xlabels=False  # Hide for the second subplot
    )
    # 3) Subplot for Group 30-36 (show xlabels)
    create_violin_subplot(
        ax=axes[2],
        data=group2_data,
        scenarios=plot_scenarios,
        group_name="Least steerable datsets (group 30-36)",
        show_xlabels=True   # Show only for the last subplot
    )
    
    # Adjust spacing so the title and subplots are nicely laid out
    plt.tight_layout(rect=[0, 0, 1, 1.03])
    
    out_file = os.path.join(output_dir, "three_subplots_vertical.pdf")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3-subplot figure to: {out_file}")


if __name__ == "__main__":
    main()
