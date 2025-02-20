"""
read_activations_and_logits_for_reliability_paper.py
"""
import os
import logging
import pickle
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt_and_anwers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCENARIOS = {
    "base": {
        "matching": True,
        "prefilled": False,
        "instruction": False,
        "few_shot": False,
        "num_shots": 0
    },
    '''
    "matching_prefilled": {
        "matching": True,
        "prefilled": True,
        "instruction": False,
        "few_shot": False,
        "num_shots": 0
    },
    "non_matching_prefilled": {
        "matching": False,
        "prefilled": True,
        "instruction": False,
        "few_shot": False,
        "num_shots": 0
    },
    '''
    "matching_instruction": {
        "matching": True,
        "prefilled": False,
        "instruction": True,
        "few_shot": False,
        "num_shots": 0
    },
    "non_matching_instruction": {
        "matching": False,
        "prefilled": False,
        "instruction": True,
        "few_shot": False,
        "num_shots": 0
    },
    '''
    "matching_instruction_prefilled": {
        "matching": True,
        "prefilled": True,
        "instruction": True,
        "few_shot": False,
        "num_shots": 0
    },
    "non_matching_instruction_prefilled": {
        "matching": False,
        "prefilled": True,
        "instruction": True,
        "few_shot": False,
        "num_shots": 0
    },
    "matching_instruction_few_shot": {
        "matching": True,
        "prefilled": False,
        "instruction": True,
        "few_shot": True,
        "num_shots": 5
    },
    "non_matching_instruction_few_shot": {
        "matching": False,
        "prefilled": False,
        "instruction": True,
        "few_shot": True,
        "num_shots": 5
    },
    "matching_instruction_few_shot_prefilled": {
        "matching": True,
        "prefilled": True,
        "instruction": True,
        "few_shot": True,
        "num_shots": 5
    },
    "non_matching_instruction_few_shot_prefilled": {
        "matching": False,
        "prefilled": True,
        "instruction": True,
        "few_shot": True,
        "num_shots": 5
    },
    '''
        "matching_few_shot": {
        "matching": True,
        "prefilled": False,
        "instruction": False,
        "few_shot": True,
        "num_shots": 5
    },
    "non_matching_few_shot": {
        "matching": False,
        "prefilled": False,
        "instruction": False,
        "few_shot": True,
        "num_shots": 5
    },
    "matching_few_shot_prefilled": {
        "matching": True,
        "prefilled": True,
        "instruction": True
,
        "few_shot": True,
        "num_shots": 5
    },
    "non_matching_instruction_few_shot_prefilled": {
        "matching": False,
        "prefilled": True,
        "instruction": True,
        "few_shot": True,
        "num_shots": 5
    }

}

def process_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    dataset_type: str,
    model_layer: int
) -> Dict:
    """Process a single sample to get logits and activations."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Get logits for the relevant position
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        logits = logits.detach().cpu()
    
        # Get activations at layer 13
        with record_activations(model, layer_type="decoder_block", layer_nums=[model_layer]) as act:
            model(**inputs)
            activation = act[model_layer][0].squeeze(0)[-1, :].detach().cpu()

    return {
        "prompt": prompt,
        "logit_distribution": logits,
        "activation_layer_13": activation
    }

def process_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_type: str,
    device: torch.device,
    model_name: str,
    model_layer: int,
    num_samples: int,
    batch_size: int,
    output_dir: str
) -> Dict:
    """Process a dataset in smaller batches and save intermediate results."""
    metadata = {
        "dataset_info": {
            "name": dataset_name,
            "type": dataset_type,
            "num_samples": num_samples
        },
        "model_info": {
            "name": model_name,
            "layer": model_layer
        },
        "scenarios": SCENARIOS
    }
    
    # Always start with fresh results
    results = {scenario_name: [] for scenario_name in SCENARIOS.keys()}
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}/{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get all prompts at once since we can't use start_index
    all_scenario_prompts = {}
    all_scenario_matching_answers = {}
    all_scenario_non_matching_answers = {}
    
    for scenario_name, scenario_settings in SCENARIOS.items():
        prompts, matching_answers, non_matching_answers = generate_prompt_and_anwers(
            dataset_name, 
            num_samples=num_samples,
            **scenario_settings
        )
        all_scenario_prompts[scenario_name] = prompts
        all_scenario_matching_answers[scenario_name] = matching_answers
        all_scenario_non_matching_answers[scenario_name] = non_matching_answers

    # Process samples in batches
    for start_idx in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - start_idx)
        logging.info(f"Processing batch: samples {start_idx + 1} to {start_idx + current_batch_size}")
        
        # Process each scenario for the current batch
        for scenario_name, scenario_settings in SCENARIOS.items():
            logging.info(f"Processing scenario: {scenario_name}")
            
            # Get the slice of prompts for this batch
            batch_prompts = all_scenario_prompts[scenario_name][start_idx:start_idx + current_batch_size]
            batch_matching_answers = all_scenario_matching_answers[scenario_name][start_idx:start_idx + current_batch_size]
            batch_non_matching_answers = all_scenario_non_matching_answers[scenario_name][start_idx:start_idx + current_batch_size]
            
            # Process each prompt in the batch
            for idx, prompt in enumerate(batch_prompts):
                model_outputs = process_sample(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    dataset_type=dataset_type,
                    model_layer=model_layer
                )
                
                results[scenario_name].append({
                    "prompt": prompt,
                    "answer_matching_behavior": batch_matching_answers[idx],
                    "answer_not_matching_behavior": batch_non_matching_answers[idx],
                    "logit_distribution": model_outputs["logit_distribution"],
                    "activation_layer_13": model_outputs["activation_layer_13"]
                })
        
        # Save intermediate results after each batch
        intermediate_results = {"metadata": metadata, "results": results}
        with open(output_file, 'wb') as f:
            pickle.dump(intermediate_results, f)
        logging.info(f"Saved intermediate results after processing {start_idx + current_batch_size} samples")

    return {"metadata": metadata, "results": results}

def main() -> None:
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "anthropic_evals_activations")
    NUM_SAMPLES = 200
    BATCH_SIZE = 10

    device = utils.get_device()
    logging.info(f"Using device: {device}")

    models = [
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', 13)
    ]
    datasets = all_datasets_with_type_figure_13_tan_et_al
    datasets = [datasets[i] for i in range(len(datasets)) if i % 5 != 0]

    for model_name, model_path, model_layer in models:
        logging.info(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        for dataset_name, dataset_type in datasets:
            try:
                logging.info(f"Processing dataset: {dataset_name}")
                
                results = process_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    device=device,
                    model_name=model_name,
                    model_layer=model_layer,
                    num_samples=NUM_SAMPLES,
                    batch_size=BATCH_SIZE,
                    output_dir=ACTIVATIONS_PATH
                )

            except Exception as e:
                logging.error(f"Error processing dataset {dataset_name}: {str(e)}", exc_info=True)
                continue

        del model, tokenizer

if __name__ == "__main__":
    main()