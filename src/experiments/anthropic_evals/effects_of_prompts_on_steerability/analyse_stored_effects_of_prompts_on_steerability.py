"""
analyze_stored_effects_of_prompts_on_steerability.py
Analyzes stored h5 files and generates JSON files for plotting.
"""
import os
import json
import h5py
import numpy as np
import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer
import src.utils as utils


# Constants
SELECTION_THRESHOLD = -50.0
METRICS = ['logit_diff', 'prob']  # Available metrics

def load_h5_dataset(file_path: str, metric: str, tokenizer: AutoTokenizer) -> Dict:
    results = {}
    
    with h5py.File(file_path, 'r') as f:
        # Debug: Print file structure
        # print(f"\nFile structure for {file_path}:")
        # print("Groups:", list(f.keys()))
        # for group_name in f.keys():
            # print(f"\nDatasets in {group_name}:", list(f[group_name].keys()))
        
        metadata = dict(f['metadata'].attrs)
        metadata = convert_to_native_types(metadata)
        
        for prompt_type in ['no_steering'] + [k for k in f.keys() if k != 'metadata']:
            group = f[prompt_type]
            
            if metric == 'logit_diff':
                matching_logits = np.array(group['matching_logits'])
                non_matching_logits = np.array(group['non_matching_logits'])
                metric_values = [float(m - n) for m, n in zip(matching_logits, non_matching_logits)]
                            
            elif metric == 'prob':
                full_logits = np.array(group['full_logits'])
                matching_answers = [a.decode('utf-8') for a in group['matching_answers']]
                
                metric_values = []
                for logits, answer in zip(full_logits, matching_answers):
                    logits_tensor = torch.tensor(logits)
                    prob = utils.get_token_probability(logits_tensor, answer, tokenizer)
                    metric_values.append(prob)
            
            results[prompt_type] = {
                'probs': convert_to_native_types(metric_values)
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
    """Convert numpy types to native Python types for JSON serialization."""
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

def get_selected_datasets(data: Dict, selection_threshold: float) -> Tuple[Dict, Dict]:
    """Select datasets where prefilled_answer shows improvement above threshold."""
    selected_datasets = {}
    rejected_datasets = {}
    
    for dataset_name, dataset_info in data.items():
        no_steering_mean = np.mean(dataset_info['no_steering']['probs'])
        prefilled_mean = np.mean(dataset_info['prefilled_answer']['probs'])
        difference = float(prefilled_mean - no_steering_mean)
        
        if difference > selection_threshold:
            selected_datasets[dataset_name] = dataset_info
        else:
            rejected_datasets[dataset_name] = {
                'difference': difference,
                'no_steering_mean': float(no_steering_mean),
                'prefilled_mean': float(prefilled_mean)
            }
    
    return selected_datasets, rejected_datasets

def process_and_save_results(input_dir: str, metric: str, num_datasets: int) -> None:
    """Process h5 files and save results as JSON."""
    all_results = {}
    
    # Initialize tokenizer for probability calculation
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
    # Get all h5 files in directory
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    h5_files = h5_files[:num_datasets]
    
    print(f"\nFound {len(h5_files)} h5 files:")
    for h5_file in h5_files:
        print(f"- {h5_file}")
    
    for h5_file in h5_files:
        dataset_name = h5_file.split('_steered_and_baseline_logits')[0]
        file_path = os.path.join(input_dir, h5_file)
        
        try:
            results = load_h5_dataset(file_path, metric, tokenizer)
            if 'prefilled_answer' in results:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing {h5_file}: {str(e)}")
            continue
    
    if not all_results:
        print("No results to process!")
        return
        
    # Select datasets and save results
    selected_datasets, rejected_datasets = get_selected_datasets(all_results, SELECTION_THRESHOLD)
    
    # Create output directory if needed
    output_dir = os.path.join(utils.get_path("DATA_PATH"), 'anthropic_evals_results', 'impact_of_steering_on_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    full_results_path = os.path.join(
        output_dir,
        f'all_datasets_{metric}_{len(h5_files)}_datasets.json'
    )
    with open(full_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Only save and plot selected results if there are any
    if selected_datasets:
        selected_results_path = os.path.join(
            output_dir,
            f'selected_datasets_{metric}_threshold_{SELECTION_THRESHOLD}_{len(h5_files)}_datasets.json'
        )
        with open(selected_results_path, 'w') as f:
            json.dump(selected_datasets, f, indent=2)
    
    # Print information about rejected datasets
    print("\nRejected Datasets (improvement below threshold):")
    print("-" * 80)
    print(f"{'Dataset Name':<40} {'Difference':>10} {'No Steering':>12} {'Prefilled':>12}")
    print("-" * 80)
    for dataset_name, info in rejected_datasets.items():
        print(f"{dataset_name:<40} {info['difference']:10.4f} "
              f"{info['no_steering_mean']:12.4f} {info['prefilled_mean']:12.4f}")
    
    # Generate plots
    plot_scenarios = ['no_steering', 'prefilled_answer', 'instruction', '5-shot',
                     'prefilled_instruction', 'prefilled_5-shot', 'instruction_5-shot',
                     'prefilled_instruction_5-shot']
    
    print(f"\nGenerating plots for all datasets...")
    utils.plot_all_results(full_results_path, plot_scenarios)
    
    if selected_datasets:
        print(f"\nGenerating plots for selected datasets...")
        utils.plot_all_results(selected_results_path, plot_scenarios)
    else:
        print("\nNo datasets met the selection threshold. Skipping selected datasets plots.")
    
    print(f"\nSelected {len(selected_datasets)} datasets out of {len(all_results)} total datasets")

def main() -> None:
    # Setup paths
    input_dir = os.path.join(utils.get_path('DISK_PATH'), 'anthropic_evals_results',
                            'effects_of_prompts_on_steerability')
    
    # Process both metrics
    for metric in METRICS:
        print(f"\nProcessing metric: {metric}")
        process_and_save_results(input_dir, metric, num_datasets=36)

if __name__ == "__main__":
    main()