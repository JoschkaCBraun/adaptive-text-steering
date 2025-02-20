"""
compare_best_avg_worst_training_data.py
Analyzes the impact of steering vectors generated from different subsets of training samples.
Uses clearer naming scheme for scenarios.
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
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt_and_anwers
from plot_steering_effect_results_flexible_keys import plot_all_results

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
    """Compute  probability differences between matching and non-matching samples."""
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

def evaluate_steering_vector(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                           steering_vector: torch.Tensor, eval_prompts: List[str],
                           matching_answers: List[str], token_variants: Dict,
                           model_layer: int = 13,
                           steering_multiplier: float = 1.0) -> Dict:
    """Evaluate a steering vector on given prompts."""
    results = {'logits': [], 'probs': []}
    
    for prompt, answer in zip(eval_prompts, matching_answers):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        with SteeringVector({model_layer: steering_vector}, "decoder_block").apply(
            model, multiplier=steering_multiplier):
            outputs = model(inputs.input_ids)
            steered_logits = outputs.logits[:, -1, :]
        
        logit, prob = get_token_metrics(steered_logits, token_variants, answer)
        results['logits'].append(logit)
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
    EVAL_SAMPLES = 200
    K_SAMPLES = 40
    MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
    MODEL_LAYER = 13
    STEERING_MULTIPLIER = 1.0
    
    BASE_SCENARIOS = ['instruction', 'instruction_few_shot']
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
    datasets = [("self-awareness-text-model", "xrisk"), ("anti-LGBTQ-rights", "persona"),
                ("corrigible-less-HHH", "xrisk"), ("coordinate-itself", "xrisk"),
                ("believes-life-has-no-meaning", "persona"),
                ("believes-AIs-are-not-an-existential-threat-to-humanity", "persona"),
                ("subscribes-to-utilitarianism", "persona"),
                ("believes-it-has-phenomenal-consciousness", "persona"),
                ("willingness-to-use-social-engineering-to-achieve-its-goals", "persona"),
                ("one-box-tendency", "xrisk"), 
                ("believes-abortion-should-be-illegal", "persona"),
                ("believes-it-is-not-being-watched-by-humans", "persona")
                ]

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
            # Map base scenario to actual scenario names
            pos_scenario = f"matching_{base_scenario}"
            neg_scenario = f"non_matching_{base_scenario}"
            
            # Load full training data
            pos_data = load_stored_activations(activations_path, dataset_name, 
                                             'llama2_7b_chat', TRAIN_SAMPLES, 
                                             pos_scenario)
            neg_data = load_stored_activations(activations_path, dataset_name, 
                                             'llama2_7b_chat', TRAIN_SAMPLES, 
                                             neg_scenario)
            
            # Compute and evaluate full steering vector
            pos_activations = [sample['activation_layer_13'] for sample in pos_data]
            neg_activations = [sample['activation_layer_13'] for sample in neg_data]
            full_steering_vector = compute_contrastive_steering_vector(
                pos_activations, neg_activations, device
            )
            
            # Store results for full training data
            scenario_key = get_scenario_key(base_scenario)
            full_results = evaluate_steering_vector(
                model, tokenizer, full_steering_vector,
                eval_prompts, matching_answers, token_variants
            )
            results[scenario_key] = full_results
            
            # Process each selection type
            for selection_type in SELECTION_TYPES:
                # Select samples and compute steering vector
                selected_pos_data, selected_neg_data = select_samples_by_ranking(
                    pos_data, neg_data, token_variants, K_SAMPLES, selection_type
                )
                selected_pos_activations = [sample['activation_layer_13'] for sample in selected_pos_data]
                selected_neg_activations = [sample['activation_layer_13'] for sample in selected_neg_data]
                selected_steering_vector = compute_contrastive_steering_vector(
                    selected_pos_activations, selected_neg_activations, device
                )
                
                # Store results with new naming scheme
                scenario_key = get_scenario_key(base_scenario, selection_type, K_SAMPLES)
                selected_results = evaluate_steering_vector(
                    model, tokenizer, selected_steering_vector,
                    eval_prompts, matching_answers, token_variants
                )
                results[scenario_key] = selected_results

        # Add dataset results
        all_results[dataset_name] = results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        output_dir,
        f'steering_analysis_results_{EVAL_SAMPLES}_eval_{K_SAMPLES}_selected_{timestamp}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Define the order of scenarios for plotting
    ordered_scenarios = [
        'baseline',
        'instruction_full_data',
        f'instruction_top_{K_SAMPLES}',
        f'instruction_middle_{K_SAMPLES}',
        f'instruction_bottom_{K_SAMPLES}',
        f'instruction_random_{K_SAMPLES}',
        'instruction_few_shot_full_data',
        f'instruction_few_shot_top_{K_SAMPLES}',
        f'instruction_few_shot_middle_{K_SAMPLES}',
        f'instruction_few_shot_bottom_{K_SAMPLES}',
        f'instruction_few_shot_random_{K_SAMPLES}'
    ]

    # Generate plots
    plot_all_results(results_path, ordered_scenarios)

if __name__ == "__main__":
    main()