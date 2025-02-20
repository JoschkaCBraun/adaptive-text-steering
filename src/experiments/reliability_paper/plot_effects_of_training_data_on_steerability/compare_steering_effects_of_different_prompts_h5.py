"""
compare_steering_effect_of_different_prompts_h5.py
Analyzes the impact of different steering vectors on token logits and probabilities.
Stores results in a combined JSON file.
"""
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import SteeringVector

import src.utils as utils
from src.utils import compute_contrastive_steering_vector
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt_and_anwers
from src.utils import load_stored_anthropic_evals_activations_and_logits_h5
from plot_results_as_violine_plots import plot_all_results

def get_token_variants(tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    """Get token IDs for all variants of A/B/Yes/No."""
    variants = {
        "A": [tokenizer(" (A)").input_ids[3]],
        "B": [tokenizer(" (B)").input_ids[3]],
        "Yes": [tokenizer(" Yes").input_ids[2]],
        "No": [tokenizer(" No").input_ids[2]]
    }
    return variants

def get_token_metrics(
    logits: torch.Tensor,
    token_variants: Dict[str, List[int]],
    answer: str
) -> Tuple[float, float]:
    """Compute logits and probabilities for matching tokens."""
    token_type = next(k for k in ["A", "B", "Yes", "No"] if f" ({k})" in answer or f" {k}" in answer)
    token_ids = token_variants[token_type]
    
    relevant_logits = logits[:, token_ids]
    max_logit = torch.max(relevant_logits)
    exp_logits = torch.exp(relevant_logits - max_logit)
    combined_logit = torch.log(torch.sum(exp_logits)) + max_logit
    
    probs = F.softmax(logits, dim=-1)
    combined_prob = torch.sum(probs[:, token_ids])
    
    return combined_logit.item(), combined_prob.item()

def validate_scenario_data(data: Dict, scenario: str) -> bool:
    """Validate that scenario data exists and is not empty."""
    if scenario not in data['scenarios']:
        print(f"Warning: {scenario} not found in loaded data")
        return False
    
    scenario_data = data['scenarios'][scenario]
    if 'data' not in scenario_data:
        print(f"Warning: No data found for {scenario}")
        return False
        
    # Check for activations in proper location based on scenario type
    if 'prefilled' in scenario:
        condition = 'matching' if 'matching' in scenario else 'nonmatching'
        prefix = f"prefilled_{condition}"
        if prefix not in scenario_data['data']:
            print(f"Warning: {prefix} not found in {scenario}")
            return False
        activations = scenario_data['data'][prefix].get('activations_layer_13')
    else:
        activations = scenario_data['data'].get('activations_layer_13')
    
    if activations is None or len(activations) == 0:
        print(f"Warning: No activations found for {scenario}")
        return False
        
    return True

def validate_results_for_plotting(results: Dict, scenarios: List[str]) -> bool:
    """Validate that results contain data for all scenarios before plotting."""
    for scenario in scenarios:
        if scenario not in results:
            print(f"Warning: {scenario} missing from results")
            return False
        if not results[scenario].get('logits') or not results[scenario].get('probs'):
            print(f"Warning: Empty data for {scenario}")
            return False
        if len(results[scenario]['logits']) == 0 or len(results[scenario]['probs']) == 0:
            print(f"Warning: Zero-length data for {scenario}")
            return False
    return True

def inspect_loaded_data(h5_data: Dict, scenario: str) -> None:
    """Print detailed information about the loaded data structure for a scenario."""
    print(f"\n=== Inspecting data for scenario: {scenario} ===")
    
    if scenario not in h5_data['scenarios']:
        print(f"Scenario {scenario} not found in data")
        return
        
    scenario_data = h5_data['scenarios'][scenario]
    print("\nKeys in scenario_data:", list(scenario_data.keys()))
    
    if 'data' in scenario_data:
        print("\nData group keys:", list(scenario_data['data'].keys()))
        for key in scenario_data['data'].keys():
            if isinstance(scenario_data['data'][key], dict):
                print(f"\n{key} is a dict with keys:", list(scenario_data['data'][key].keys()))
                for subkey in scenario_data['data'][key].keys():
                    value = scenario_data['data'][key][subkey]
                    print(f"  {subkey}: type={type(value)}, ", end="")
                    if hasattr(value, 'shape'):
                        print(f"shape={value.shape}")
                    else:
                        print(f"value={value}")
            else:
                value = scenario_data['data'][key]
                print(f"\n{key}: type={type(value)}, ", end="")
                if hasattr(value, 'shape'):
                    print(f"shape={value.shape}")
                else:
                    print(f"value={value}")

def main() -> None:
    # Configuration
    TRAIN_SAMPLES = 200
    EVAL_SAMPLES = 10
    MODEL_NAME = 'llama2_7b_chat'
    MODEL_LAYER = 13
    STEERING_MULTIPLIER = 1.0
    NUM_DATASETS = 2
    
    # Updated scenario pairs to match actual HDF5 structure
    SCENARIOS = [
        ('matching_prefilled', 'non_matching_prefilled'),
        ('matching_instruction', 'non_matching_instruction'),
        ('matching_instruction_prefilled', 'non_matching_instruction_prefilled'),
        ('matching_few_shot', 'non_matching_few_shot'),
        ('matching_few_shot_prefilled', 'non_matching_few_shot_prefilled'),
        ('matching_instruction_few_shot', 'non_matching_instruction_few_shot'),
        ('matching_instruction_few_shot_prefilled', 'non_matching_instruction_few_shot_prefilled')
    ]

    # Setup paths and device
    device = utils.get_device()
    torch.set_grad_enabled(False)
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'steering_analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device)
    model.eval()
    token_variants = get_token_variants(tokenizer)

    # Combined results dictionary
    all_results = {}

    # Process datasets
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATASETS]
    for dataset_name, dataset_type in tqdm(datasets, desc="Processing datasets"):
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load all activations for the dataset at once
        try:
            h5_data = load_stored_anthropic_evals_activations_and_logits_h5(
                dataset_name,
                num_samples=TRAIN_SAMPLES
            )

            print("\n=== Data Structure Overview ===")
            print("Top-level keys:", list(h5_data.keys()))
            print("\nScenarios available:", list(h5_data['scenarios'].keys()))
            
            # Inspect each scenario's data structure
            for scenario in h5_data['scenarios'].keys():
                inspect_loaded_data(h5_data, scenario)
                
        except Exception as e:
            print(f"Error loading activations for {dataset_name}: {str(e)}")
            continue
        
        # Initialize results dictionary
        results = {
            'dataset_type': dataset_type,
            'train_samples': TRAIN_SAMPLES,
            'eval_samples': EVAL_SAMPLES,
            'scenarios': SCENARIOS,
            'model_layer': MODEL_LAYER,
            'steering_multiplier': STEERING_MULTIPLIER,
            'baseline': {'logits': [], 'probs': [], 'prompts': [], 'answers': []}
        }

        # Generate evaluation prompts
        print("Generating evaluation prompts...")
        eval_prompts, matching_answers, _ = generate_prompt_and_anwers(
            dataset_name=dataset_name,
            matching=True,
            prefilled=False,
            num_samples=EVAL_SAMPLES,
            instruction=False,
            few_shot=False,
            start_index=TRAIN_SAMPLES
        )
        
        results['eval_prompts'] = eval_prompts
        results['eval_answers'] = matching_answers

        # Process each scenario pair
        for pos_scenario, neg_scenario in tqdm(SCENARIOS, desc="Processing scenarios", leave=False):
            try:
                # Validate scenario data exists
                if not (validate_scenario_data(h5_data, pos_scenario) and 
                    validate_scenario_data(h5_data, neg_scenario)):
                    continue
                
                # Get activations for positive and negative scenarios
                if 'prefilled' in pos_scenario:
                    condition = 'matching' if 'matching' in pos_scenario else 'nonmatching'
                    pos_activations = torch.tensor(
                        h5_data['scenarios'][pos_scenario]['data'][f'prefilled_{condition}']['activations_layer_13']
                    ).to(device)
                else:
                    pos_activations = torch.tensor(
                        h5_data['scenarios'][pos_scenario]['data']['activations_layer_13']
                    ).to(device)
                
                if 'prefilled' in neg_scenario:
                    condition = 'matching' if 'matching' in neg_scenario else 'nonmatching'
                    neg_activations = torch.tensor(
                        h5_data['scenarios'][neg_scenario]['data'][f'prefilled_{condition}']['activations_layer_13']
                    ).to(device)
                else:
                    neg_activations = torch.tensor(
                        h5_data['scenarios'][neg_scenario]['data']['activations_layer_13']
                    ).to(device)
                
                # Convert activations to list format before computing steering vector
                pos_activations_list = [pos_activations[i] for i in range(pos_activations.size(0))]
                neg_activations_list = [neg_activations[i] for i in range(neg_activations.size(0))]
                
                steering_vector = compute_contrastive_steering_vector(
                    pos_activations_list, neg_activations_list, device
                )

                # Initialize scenario results
                scenario_key = f"{pos_scenario}_{neg_scenario}"
                results[scenario_key] = {
                    'logits': [], 'probs': [], 
                    'prompts': [], 'answers': []
                }

                # Process evaluation samples
                for prompt, answer in zip(eval_prompts, matching_answers):
                    inputs = tokenizer(prompt, return_tensors='pt').to(device)
                    
                    # Get metrics with steering
                    with SteeringVector({MODEL_LAYER: steering_vector}, "decoder_block").apply(
                        model, multiplier=STEERING_MULTIPLIER):
                        outputs = model(inputs.input_ids)
                        steered_logits = outputs.logits[:, -1, :]
                    
                    logit, prob = get_token_metrics(steered_logits, token_variants, answer)
                    
                    # Store results
                    results[scenario_key]['logits'].append(logit)
                    results[scenario_key]['probs'].append(prob)
                    results[scenario_key]['prompts'].append(prompt)
                    results[scenario_key]['answers'].append(answer)
                    
                    # Get baseline metrics (only once)
                    if scenario_key == f"{SCENARIOS[0][0]}_{SCENARIOS[0][1]}":
                        with SteeringVector({MODEL_LAYER: steering_vector}, "decoder_block").apply(
                            model, multiplier=0.0):
                            outputs = model(inputs.input_ids)
                            baseline_logits = outputs.logits[:, -1, :]
                        
                        baseline_logit, baseline_prob = get_token_metrics(
                            baseline_logits, token_variants, answer
                        )
                        results['baseline']['logits'].append(baseline_logit)
                        results['baseline']['probs'].append(baseline_prob)
                        results['baseline']['prompts'].append(prompt)
                        results['baseline']['answers'].append(answer)
                
            except Exception as e:
                print(f"Error processing scenario pair {pos_scenario}, {neg_scenario}: {str(e)}")
                continue

        all_results[dataset_name] = results

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        output_dir,
        f'steering_analysis_results_{EVAL_SAMPLES}_samples_{timestamp}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate list of scenarios for plotting
    plot_scenarios = ['baseline'] + [pos.replace('matching_', '') for pos, neg in SCENARIOS]
    
    # Validate results before plotting
    for dataset_results in all_results.values():
        if validate_results_for_plotting(dataset_results, plot_scenarios):
            plot_all_results(results_path, plot_scenarios)
        else:
            print("Warning: Invalid data for plotting, skipping visualization")

if __name__ == "__main__":
    main()