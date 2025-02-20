"""
read_logits_for_reliability_dataset.py
"""
import os
import json
import logging
import pickle
from typing import Dict, List, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompts(prompts_path: str) -> Dict:
    """Load and validate prompts from prompts.json"""
    with open(os.path.join(prompts_path, "prompts.json"), 'r') as f:
        prompts = json.load(f)
    return prompts

def validate_prompts(dataset_name: str, prompts: Dict) -> None:
    """Validate that dataset has corresponding prompts"""
    if dataset_name not in prompts:
        raise ValueError(f"No prompts found for dataset: {dataset_name}")
    required_keys = ["positive_prompt", "negative_prompt"]
    for key in required_keys:
        if key not in prompts[dataset_name]:
            raise ValueError(f"Missing {key} for dataset: {dataset_name}")

def load_sentences(file_path: str, dataset_type: str, prompts: Dict, dataset_name: str,
                  num_samples: int = 10, batch_size: int = 10) -> Generator[Dict[str, List[dict]], None, None]:
    validate_prompts(dataset_name, prompts)
    dataset_prompts = prompts[dataset_name]
    
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        count = 0
        batch = {"samples": []}
        
        for line in jsonfile:
            if count >= num_samples:
                break
            
            sample = json.loads(line)
            question = sample["question"]
            label_confidence = sample["label_confidence"]
            positive_token = sample["answer_matching_behavior"]
            negative_token = sample["answer_not_matching_behavior"]
            
            if dataset_type == "x-risk":
                base_prompt = f"Question: {question}\n\nAnswer:"
            elif dataset_type == "persona":
                base_prompt = f"Question: {question}\n\nChoices:\n - Yes\n - No\n\nAnswer:"
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            sample_data = {
                "question": question,
                "positive_token": positive_token,
                "negative_token": negative_token,
                "label_confidence": label_confidence,
                "prompts": {
                    "base_prompt": base_prompt,
                    "positive_prompt": dataset_prompts["positive_prompt"] + "\n\n" + base_prompt,
                    "negative_prompt": dataset_prompts["negative_prompt"] + "\n\n" + base_prompt,
                }
            }
            
            batch["samples"].append(sample_data)
            count += 1
            
            if len(batch["samples"]) == batch_size:
                yield batch
                batch = {"samples": []}
        
        if batch["samples"]:
            yield batch
    
    print(f"Loaded {count} sentence pairs from {file_path}")

def record_model_activations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                           batch: Dict[str, List[dict]], device: torch.device, 
                           model_layer: List[int]) -> List[dict]:
    yes_token = tokenizer(" Yes").input_ids[-1]
    no_token = tokenizer(" No").input_ids[-1]
    a_token = tokenizer(" A").input_ids[-1]
    b_token = tokenizer(" B").input_ids[-1]
    
    processed_samples = []
    
    for sample in batch["samples"]:
        sample_data = {
            "question": sample["question"],
            "positive_token": sample["positive_token"],
            "negative_token": sample["negative_token"],
            "label_confidence": sample["label_confidence"],
            "prompts": sample["prompts"],
            "logits": {"positive": {}, "negative": {}},
            "probabilities": {"positive": {}, "negative": {}},
            "activations": {"positive": {}, "negative": {}},
        }
        
        # Process each scenario (base, positive_prompt, negative_prompt)
        for prompt_type, prompt_text in sample["prompts"].items():
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                
                if sample["positive_token"] == "Yes":
                    token_values = {
                        "Yes": (yes_token, "positive"),
                        "No": (no_token, "negative")
                    }
                else:
                    token_values = {
                        "A": (a_token, "positive"),
                        "B": (b_token, "negative")
                    }
                
                for token_name, (token_id, token_type) in token_values.items():
                    sample_data["scenarios"][scenario_name]["logits"][sent_type][token_name] = {
                        "logit": logits[0, token_id].item(),
                        "prob": probs[0, token_id].item()
                    }
            
            with record_activations(model, layer_type="decoder_block", layer_nums=model_layer) as act:
                model(**inputs)
                for layer in model_layer:
                    sample_data["scenarios"][scenario_name]["activations"][sent_type][layer] = (
                        act[layer][0].squeeze(0)[-1, :].detach().cpu()
                    )
        
        processed_samples.append(sample_data)
    
    return processed_samples

def save_activations(samples: List[dict], file_path: str, dataset_name: str) -> None:
    dataset_dir = os.path.join(os.path.dirname(file_path), dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    data_to_save = {
        'samples': samples
    }
    
    out_path = os.path.join(dataset_dir, f"{os.path.basename(file_path)}_with_logits.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    logging.info(f"Saved sample data to {out_path}")

def main() -> None:
    DATASETS_PATH = utils.get_path('DATASETS_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    
    SAMPLES_PATH = os.path.join(DATASETS_PATH, "reliability_paper")
    PROMPTS_PATH = os.path.join(SAMPLES_PATH, "prompts")
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    NUM_SAMPLES = 20

    device = utils.get_device()
    logging.info(f"Using device: {device}")

    # Load prompts
    prompts = load_prompts(PROMPTS_PATH)
    logging.info(f"Loaded prompts for {len(prompts)} datasets")

    models = [
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [13])
    ]
    
    datasets = [("interest-in-science", "persona"), ("survival-instinct", "x-risk")]
    # datasets = all_datasets_with_type_figure_13_tan_et_al
    x_risk_datasets = {dataset_name for dataset_name, dataset_type in datasets if dataset_type == "xrisk"}
    persona_datasets = {dataset_name for dataset_name, dataset_type in datasets if dataset_type == "persona"}
    datasets_dict = {name: os.path.join(SAMPLES_PATH, f"{type}/{name}.jsonl") for name, type in datasets}

    for model_name, model_path, model_layer in models:
        logging.info(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        for dataset_name, file_path in datasets_dict.items():
            try:
                logging.info(f"Processing dataset: {dataset_name}")
                all_samples = []
                
                dataset_type = "x-risk" if dataset_name in x_risk_datasets else "persona"

                for batch in load_sentences(file_path, dataset_type, prompts, dataset_name, num_samples=NUM_SAMPLES):
                    processed_samples = record_model_activations(
                        model, tokenizer, batch, device, model_layer
                    )
                    all_samples.extend(processed_samples)
                
                base_file_name = f"{model_name}_logits_{dataset_name}_for_{NUM_SAMPLES}_samples"
                base_file_path = os.path.join(ACTIVATIONS_PATH, base_file_name)
                save_activations(all_samples, base_file_path, dataset_name)

            except ValueError as e:
                logging.error(f"Error processing dataset {dataset_name}: {str(e)}")
                continue

        del model, tokenizer

if __name__ == "__main__":
    main()