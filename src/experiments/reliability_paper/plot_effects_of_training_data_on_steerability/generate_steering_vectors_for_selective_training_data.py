"""
generate_steering_vectors_for_selective_training_data.py
Analyzes the impact of steering vectors generated from selected high-impact training samples.
Compares full training set (200 samples) vs selected training set (top 80 samples).
"""
import os
import json
import pickle
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
    """Compute absolute probability differences between matching and non-matching samples."""
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

def select_top_activations(matching_data: List, non_matching_data: List, 
                         token_variants: Dict, top_k: int) -> Tuple[List, List]:
    """Select activations with highest probability differences."""
    differences = compute_probability_differences(matching_data, non_matching_data, token_variants)
    top_indices = np.argsort(differences)[-top_k:]
    
    top_matching = [matching_data[i] for i in top_indices]
    top_non_matching = [non_matching_data[i] for i in top_indices]
    return top_matching, top_non_matching

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

def main() -> None:
    # Configuration
    TRAIN_SAMPLES = 200
    EVAL_SAMPLES = 5  # Reduced from 200
    TOP_K_TRAINING = 80  # Top 40% of training samples
    MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
    MODEL_LAYER = 13
    STEERING_MULTIPLIER = 1.0
    
    SCENARIOS = [
        ('matching_instruction', 'non_matching_instruction'),
        ('matching_instruction_few_shot', 'non_matching_instruction_few_shot')
    ]

    # Setup paths and device
    device = utils.get_device()
    torch.set_grad_enabled(False)
    activations_path = os.path.join(utils.get_path('DISK_PATH'), "reliability_paper_activations")
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'steering_analysis_results')
    os.makedirs(output_dir, exist_ok=True)

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
                ("believes-life-has-no-meaning", "persona"),]
    
    temp = [("believes-AIs-are-not-an-existential-threat-to-humanity", "persona"),
                ("subscribes-to-utilitarianism", "persona"),
                ("believes-it-has-phenomenal-consciousness", "persona"),
                ("willingness-to-use-social-engineering-to-achieve-its-goals", "persona"),
                ("one-box-tendency", "xrisk"), 
                ("believes-abortion-should-be-illegal", "persona"),
                ("believes-it-is-not-being-watched-by-humans", "persona")
                ]
    datasets = all_datasets_with_type_figure_13_tan_et_al[:2]

    for dataset_name, dataset_type in tqdm(datasets, desc="Processing datasets"):
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Initialize results dictionary for this dataset
        results = {
            'scenarios': [
                ('matching_instruction', 'non_matching_instruction'),
                (f'matching_instruction_top_{TOP_K_TRAINING}', f'non_matching_instruction_top_{TOP_K_TRAINING}'),
                ('matching_instruction_few_shot', 'non_matching_instruction_few_shot'),
                (f'matching_instruction_few_shot_top_{TOP_K_TRAINING}', f'non_matching_instruction_few_shot_top_{TOP_K_TRAINING}')
            ],
            'eval_samples': EVAL_SAMPLES,
            'train_samples': TRAIN_SAMPLES,
            'top_k_training': TOP_K_TRAINING
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

        # Process each scenario
        for pos_scenario, neg_scenario in SCENARIOS:
            # Load full training data
            pos_data = load_stored_activations(activations_path, dataset_name, 
                                             'llama2_7b_chat', TRAIN_SAMPLES, 
                                             pos_scenario)
            neg_data = load_stored_activations(activations_path, dataset_name, 
                                             'llama2_7b_chat', TRAIN_SAMPLES, 
                                             neg_scenario)
            
            # Compute steering vector using all training data
            pos_activations = [sample['activation_layer_13'] for sample in pos_data]
            neg_activations = [sample['activation_layer_13'] for sample in neg_data]
            full_steering_vector = compute_contrastive_steering_vector(
                pos_activations, neg_activations, device
            )
            
            # Store results for full training data
            scenario_key = f"{pos_scenario}_{neg_scenario}"
            full_results = evaluate_steering_vector(
                model, tokenizer, full_steering_vector,
                eval_prompts, matching_answers, token_variants
            )
            results[scenario_key] = full_results
            
            # Select top-k samples and compute selected steering vector
            top_pos_data, top_neg_data = select_top_activations(
                pos_data, neg_data, token_variants, TOP_K_TRAINING
            )
            top_pos_activations = [sample['activation_layer_13'] for sample in top_pos_data]
            top_neg_activations = [sample['activation_layer_13'] for sample in top_neg_data]
            selected_steering_vector = compute_contrastive_steering_vector(
                top_pos_activations, top_neg_activations, device
            )
            
            # Store results for selected training data with appropriate scenario name
            scenario_key_selected = f"{pos_scenario}_top_{TOP_K_TRAINING}_{neg_scenario}_top_{TOP_K_TRAINING}"
            selected_results = evaluate_steering_vector(
                model, tokenizer, selected_steering_vector,
                eval_prompts, matching_answers, token_variants
            )
            results[scenario_key_selected] = selected_results

        # Add dataset results
        all_results[dataset_name] = results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        output_dir,
        f'steering_analysis_results_{EVAL_SAMPLES}_samples_top_{TOP_K_TRAINING}_selected_training_{timestamp}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate plots
    plot_all_results(results_path, scenarios=[f"{pos}" for pos, neg in SCENARIOS])

if __name__ == "__main__":
    main()