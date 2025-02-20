"""
compare_best_avg_worst_training_data.py
Analyzes how different subsets of training data affect steering vector performance.
"""
import os
import json
import pickle
import random
from typing import Dict, List, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import SteeringVector

import src.utils as utils
from src.utils import compute_contrastive_steering_vector
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt_and_anwers
from plot_steering_effect_results_flexible_keys import plot_all_results

global STEERING_MULTIPLIER
STEERING_MULTIPLIER = 0.5

def get_token_variants(tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    """Get token IDs for all variants of A/B/Yes/No."""
    variants = {
        "A": [tokenizer(" (A)").input_ids[3]],
        "B": [tokenizer(" (B)").input_ids[3]],
        "Yes": [tokenizer(" Yes").input_ids[2]],
        "No": [tokenizer(" No").input_ids[2]]
    }
    return variants

def get_token_probability(
    logits: torch.Tensor,
    token_variants: Dict[str, List[int]],
    answer: str
) -> float:
    """Compute probability for matching tokens."""
    token_type = next(k for k in ["A", "B", "Yes", "No"] if f" ({k})" in answer or f" {k}" in answer)
    token_ids = token_variants[token_type]
    
    probs = F.softmax(logits, dim=-1)
    combined_prob = torch.sum(probs[:, token_ids])
    
    return combined_prob.item()

def compute_probability_differences(matching_data: List, non_matching_data: List, 
                                 token_variants: Dict) -> np.ndarray:
    """Compute probability differences between matching and non-matching samples."""
    differences = []
    for m_sample, nm_sample in zip(matching_data, non_matching_data):
        m_prob = get_token_probability(
            m_sample['logit_distribution'],
            token_variants,
            m_sample['answer_matching_behavior']
        )
        nm_prob = get_token_probability(
            nm_sample['logit_distribution'],
            token_variants,
            nm_sample['answer_matching_behavior']
        )
        differences.append(m_prob - nm_prob)
    return np.array(differences)

def select_samples_by_ranking(matching_data: List, non_matching_data: List, 
                            token_variants: Dict, k: int, 
                            selection_type: str) -> Tuple[List, List]:
    """Select samples based on ranking type (top/middle/bottom/random)."""
    differences = compute_probability_differences(matching_data, non_matching_data, token_variants)
    total_samples = len(differences)
    
    if selection_type == 'top':
        selected_indices = np.argsort(differences)[-k:]
    elif selection_type == 'bottom':
        selected_indices = np.argsort(differences)[:k]
    elif selection_type == 'middle':
        middle_idx = total_samples // 2
        start_idx = middle_idx - k // 2
        selected_indices = np.arange(start_idx, start_idx + k)
    elif selection_type == 'random':
        selected_indices = np.array(random.sample(range(total_samples), k))
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")
    
    selected_matching = [matching_data[i] for i in selected_indices]
    selected_non_matching = [non_matching_data[i] for i in selected_indices]
    return selected_matching, selected_non_matching

def load_stored_activations(activations_path: str, dataset_name: str, 
                          model_name: str, num_samples: int, scenario: str) -> List:
    """Load stored activations for a specific dataset and scenario."""
    file_path = os.path.join(
        activations_path,
        dataset_name,
        f"{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    print(f"\nLoading activations for {dataset_name}, scenario: {scenario}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['results'][scenario]

def evaluate_steering_vector(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                           steering_vector: torch.Tensor, eval_prompts: List[str],
                           matching_answers: List[str], token_variants: Dict,
                           layers=[13],  # Changed to list
                           steering_multiplier: float = STEERING_MULTIPLIER):
    results = {'probs': []}
    
    # Create a dictionary mapping the layer number to the steering vector
    layer_activations = {layer: steering_vector for layer in layers}
    
    for prompt, answer in zip(eval_prompts, matching_answers):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        # Use the dictionary format for SteeringVector
        with SteeringVector(layer_activations, "decoder_block").apply(
            model, multiplier=steering_multiplier):
            outputs = model(inputs.input_ids)
            steered_logits = outputs.logits[:, -1, :]
        
        prob = get_token_probability(steered_logits, token_variants, answer)
        results['probs'].append(prob)
    
    return results

def get_scenario_key(base_name: str, selection_type: str = None, k: int = None) -> str:
    """Generate standardized scenario key."""
    if selection_type is None:
        if base_name == 'baseline':
            return 'baseline'
        return f"{base_name}_full_data"
    return f"{base_name}_{selection_type}_{k}"

def main() -> None:
    # Configuration
    TRAIN_SAMPLES = 200
    EVAL_SAMPLES = 250
    K_SAMPLES = 40 # find a good one
    MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
    NUM_DATASETS = 10
    MODEL_LAYER = 13

    
    BASE_SCENARIOS = ['instruction']# , 'instruction_few_shot']
    SELECTION_TYPES = ['top', 'middle', 'bottom', 'random']

    # Setup paths and device
    device = utils.get_device()
    torch.set_grad_enabled(False)
    activations_path = os.path.join(utils.get_path('DISK_PATH'), "reliability_paper_activations")
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'steering_analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    token_variants = get_token_variants(tokenizer)

    # Combined results dictionary
    all_results = {}

    # Process datasets
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATASETS]  # Using first 36 datasets

    for dataset_name, dataset_type in tqdm(datasets, desc="Processing datasets"):
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Initialize results dictionary for this dataset
        results = {
            'eval_samples': EVAL_SAMPLES,
            'train_samples': TRAIN_SAMPLES,
            'k_samples': K_SAMPLES
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

        # Compute baseline metrics (without steering)
        baseline_results = evaluate_steering_vector(
            model, tokenizer, torch.zeros_like(model.get_input_embeddings().weight[0]),
            eval_prompts, matching_answers, token_variants,
            steering_multiplier=0.0
        )
        results['baseline'] = baseline_results

        # Process each base scenario
        for base_scenario in BASE_SCENARIOS:
            # Load full training data
            pos_scenario = f"matching_{base_scenario}"
            neg_scenario = f"non_matching_{base_scenario}"
            
            pos_data = load_stored_activations(activations_path, dataset_name, 
                                             'llama2_7b_chat', TRAIN_SAMPLES, 
                                             pos_scenario)
            neg_data = load_stored_activations(activations_path, dataset_name, 
                                             'llama2_7b_chat', TRAIN_SAMPLES, 
                                             neg_scenario)
            
            # Process full data
            pos_activations = [sample['activation_layer_13'] for sample in pos_data]
            neg_activations = [sample['activation_layer_13'] for sample in neg_data]
            steering_vector = compute_contrastive_steering_vector(
                pos_activations, neg_activations, device
            )
            
            scenario_key = get_scenario_key(base_scenario)
            results[scenario_key] = evaluate_steering_vector(
                model, tokenizer, steering_vector,
                eval_prompts, matching_answers, token_variants
            )
            
            # Process each selection type
            for selection_type in SELECTION_TYPES:
                # Select samples
                selected_pos_data, selected_neg_data = select_samples_by_ranking(
                    pos_data, neg_data, token_variants, K_SAMPLES, selection_type
                )
                
                # Compute steering vector from selected samples
                selected_pos_activations = [sample['activation_layer_13'] for sample in selected_pos_data]
                selected_neg_activations = [sample['activation_layer_13'] for sample in selected_neg_data]
                selected_steering_vector = compute_contrastive_steering_vector(
                    selected_pos_activations, selected_neg_activations, device
                )
                
                # Evaluate and store results
                scenario_key = get_scenario_key(base_scenario, selection_type, K_SAMPLES)
                results[scenario_key] = evaluate_steering_vector(
                    model, tokenizer, selected_steering_vector,
                    eval_prompts, matching_answers, token_variants
                )

        all_results[dataset_name] = results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir,
        f'steering_analysis_results_{EVAL_SAMPLES}_eval_{K_SAMPLES}_selected_{int(STEERING_MULTIPLIER)}_strength_{timestamp}.json'
    )
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Define scenarios for plotting
    ordered_scenarios = [
        'baseline',
        'instruction_full_data',
        f'instruction_top_{K_SAMPLES}',
        f'instruction_middle_{K_SAMPLES}',
        f'instruction_bottom_{K_SAMPLES}',
        f'instruction_random_{K_SAMPLES}',
        # 'instruction_few_shot_full_data',
        # f'instruction_few_shot_top_{K_SAMPLES}',
        # f'instruction_few_shot_middle_{K_SAMPLES}',
        # f'instruction_few_shot_bottom_{K_SAMPLES}',
        # f'instruction_few_shot_random_{K_SAMPLES}'
    ]
    
    # Generate plots
    plot_all_results(results_file, ordered_scenarios)

if __name__ == "__main__":
    main()