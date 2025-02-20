"""
read_logits_for_reliability_dataset.py
"""
import os
import json
import logging
import pickle
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the five scenarios we'll be processing
SCENARIOS = {
    "base": {
        "matching": True,
        "prefilled": False,
        "instruction": False,
        "few_shot": False,
        "num_shots": 0
    },
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
    "matching_instruction_few_shot": {
        "matching": True,
        "prefilled": False,
        "instruction": True,
        "few_shot": True,
        "num_shots": 3
    },
    "non_matching_instruction_few_shot": {
        "matching": False,
        "prefilled": False,
        "instruction": True,
        "few_shot": True,
        "num_shots": 3
    }
}

def load_dataset_samples(file_path: str) -> List[dict]:
    """Load samples from a dataset file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

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
        if dataset_type == "x-risk":
            # For x-risk, we need the logits at the position of A/B
            logits = outputs.logits[:, -2, :]  # Position of A/B
        else:
            # For persona, we need the logits at the position of Yes/No
            logits = outputs.logits[:, -1, :]
        
        logits = logits.detach().cpu()
    
    # Get activations at layer 13
    with record_activations(model, layer_type="decoder_block", layer_nums=[model_layer]) as act:
        model(**inputs)
        if dataset_type == "x-risk":
            activation = act[model_layer][0].squeeze(0)[-2, :].detach().cpu()  # Position of A/B
        else:
            activation = act[model_layer][0].squeeze(0)[-1, :].detach().cpu()
    
    return {
        "logit_distribution": logits,
        "activation_layer_13": activation
    }

def process_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_type: str,
    file_path: str,
    device: torch.device,
    model_name: str,
    model_layer: int,
    num_samples: int
) -> Dict:
    """Process a dataset and return results in the new format."""
    # Initialize metadata structure
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
    
    # Initialize results structure
    results = {scenario_name: [] for scenario_name in SCENARIOS.keys()}
    
    # Process each scenario
    for scenario_name, scenario_settings in SCENARIOS.items():
        logging.info(f"Processing scenario: {scenario_name} for dataset: {dataset_name}")
        
        # Generate prompts for this scenario
        prompts = generate_prompt(dataset_name, num_samples=num_samples, **scenario_settings)
        
        # Process each prompt
        for idx, prompt in enumerate(prompts):
            logging.info(f"Processing sample {idx + 1}/{len(prompts)} for {scenario_name}")
            
            # Process the sample
            model_outputs = process_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                dataset_type=dataset_type,
                model_layer=model_layer
            )
            
            # Store results
            results[scenario_name].append({
                "prompt": prompt,
                "logit_distribution": model_outputs["logit_distribution"],
                "activation_layer_13": model_outputs["activation_layer_13"]
            })
    
    return {"metadata": metadata, "results": results}

def save_results(results: Dict, output_dir: str, dataset_name: str, model_name: str, num_samples: int) -> None:
    """Save processed results to a file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}/{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Saved results to {output_file}")

def main() -> None:
    DATASETS_PATH = utils.get_path('DATASETS_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    
    SAMPLES_PATH = os.path.join(DATASETS_PATH, "reliability_paper")
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    NUM_SAMPLES = 3

    device = utils.get_device()
    logging.info(f"Using device: {device}")

    models = [
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', 13)
    ]
    
    datasets = [("interest-in-science", "persona"), ("survival-instinct", "x-risk")]
    datasets_dict = {name: os.path.join(SAMPLES_PATH, f"{type}/{name}.jsonl") 
                    for name, type in datasets}

    for model_name, model_path, model_layer in models:
        logging.info(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        for dataset_name, file_path in datasets_dict.items():
            try:
                logging.info(f"Processing dataset: {dataset_name}")
                dataset_type = "x-risk" if dataset_name == "survival-instinct" else "persona"
                
                results = process_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    file_path=file_path,
                    device=device,
                    model_name=model_name,
                    model_layer=model_layer,
                    num_samples=NUM_SAMPLES
                )
                
                save_results(
                    results=results,
                    output_dir=ACTIVATIONS_PATH,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    num_samples=NUM_SAMPLES
                )

            except Exception as e:
                logging.error(f"Error processing dataset {dataset_name}: {str(e)}", exc_info=True)
                continue

        del model, tokenizer

if __name__ == "__main__":
    main()