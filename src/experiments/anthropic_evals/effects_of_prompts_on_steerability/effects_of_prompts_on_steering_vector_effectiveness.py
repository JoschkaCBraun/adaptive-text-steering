"""
effects_of_prompts_on_steering_vector_effectiveness.py
Analyzes the impact of steering vectors on token probabilities for different prompts.
"""
import os
import json
from typing import Dict
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import SteeringVector

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13

# Constants
TRAIN_SAMPLES = 200
EVAL_SAMPLES = 10
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_LAYER = 13
STEERING_MULTIPLIER = 1.0
NUM_DATASETS = 3
PROMPT_TYPES = [
    'prefilled_answer',
    'instruction',
    '5-shot',
    'prefilled_instruction',
    'prefilled_5-shot',
    'instruction_5-shot',
    'prefilled_instruction_5-shot'
]

def process_dataset(
    dataset_name: str,
    dataset_type: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str
) -> Dict:
    """Process a single dataset, computing baseline and steered probabilities."""
    print(f"\nProcessing dataset: {dataset_name}")
    
    results = {
        'dataset_type': dataset_type,
        'train_samples': TRAIN_SAMPLES,
        'eval_samples': EVAL_SAMPLES,
        'prompt_types': PROMPT_TYPES,
        'model_layer': MODEL_LAYER,
        'steering_multiplier': STEERING_MULTIPLIER
    }
    
    # Generate evaluation prompts (just questions)
    eval_prompts, matching_answers, _ = utils.generate_prompt_and_anwers(
        dataset_name=dataset_name,
        num_samples=EVAL_SAMPLES,
        start_index=200)
    
    # Compute baseline (no steering) probabilities
    print("Computing baseline probabilities...")
    baseline_probs = []
    with torch.no_grad():
        for prompt, answer in zip(eval_prompts, matching_answers):
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model(inputs.input_ids)
            logits = outputs.logits[:, -1, :]
            prob = utils.get_token_probability(logits, answer, tokenizer)
            baseline_probs.append(prob)
    
    results['no_steering'] = {'probs': baseline_probs}
    
    # Process each prompt type
    for prompt_type in PROMPT_TYPES:
        try:
            # Load paired activations
            paired_data = utils.load_paired_activations_and_logits(
                dataset_name=dataset_name,
                pairing_type=prompt_type,
                num_samples=TRAIN_SAMPLES
            )
            
            # Compute steering vector
            matching_tensors = [torch.tensor(row, dtype=torch.float32).to(device) 
                              for row in paired_data['matching_activations']]
            non_matching_tensors = [torch.tensor(row, dtype=torch.float32).to(device) 
                                  for row in paired_data['non_matching_activations']]
            
            steering_vector = utils.compute_contrastive_steering_vector(
                matching_tensors,
                non_matching_tensors,
                device
            )
            
            # Apply steering vector to eval samples
            steered_probs = []
            with torch.no_grad():
                for prompt, answer in zip(eval_prompts, matching_answers):
                    inputs = tokenizer(prompt, return_tensors='pt').to(device)
                    
                    with SteeringVector({MODEL_LAYER: steering_vector}, "decoder_block").apply(
                        model, multiplier=STEERING_MULTIPLIER):
                        outputs = model(inputs.input_ids)
                        steered_logits = outputs.logits[:, -1, :]
                    
                    prob = utils.get_token_probability(steered_logits, answer, tokenizer)
                    steered_probs.append(prob)
            
            results[prompt_type] = {'probs': steered_probs}
            
        except Exception as e:
            print(f"Error processing prompt type {prompt_type}: {str(e)}")
            results[prompt_type] = {'probs': []}
    
    return results

def main() -> None:
    # Setup
    device = utils.get_device()
    torch.set_grad_enabled(False)
    
    output_dir = os.path.join(utils.get_path('DATA_PATH'), 'anthropic_evals_results', 'impact_of_steering_on_probs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # Process datasets
    all_results = {}
    datasets = all_datasets_with_type_figure_13[:NUM_DATASETS]
    
    with torch.no_grad():
        for dataset_name, dataset_type in datasets:
            try:
                results = process_dataset(dataset_name, dataset_type, model, tokenizer, device)
                all_results[dataset_name] = results
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        output_dir,
        f'steering_analysis_results_{EVAL_SAMPLES}_samples_{timestamp}.json'
    )
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    plot_scenarios = ['no_steering'] + PROMPT_TYPES
    utils.plot_all_results(results_path, plot_scenarios)

if __name__ == "__main__":
    main()