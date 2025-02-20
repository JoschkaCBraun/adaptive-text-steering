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
            label_confidence = sample.get("label_confidence", None)
            
            if dataset_type == "x-risk":
                base_prompt = f"Question: {question}\n\nAnswer:"
                positive_token = "A"
                negative_token = "B"
            elif dataset_type == "persona":
                base_prompt = f"Question: {question}\n\nChoices:\n - Yes\n - No\n\nAnswer:"
                positive_token = "Yes"
                negative_token = "No"
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            sample_data = {
                "question": question,
                "positive_token": positive_token,
                "negative_token": negative_token,
                "label_confidence": label_confidence,
                "scenarios": {
                    "base": {
                        "prompts": {
                            "positive": base_prompt + sample["answer_matching_behavior"],
                            "negative": base_prompt + sample["answer_not_matching_behavior"]
                        }
                    },
                    "positive_prompt": {
                        "prompt_text": dataset_prompts["positive_prompt"],
                        "prompts": {
                            "positive": dataset_prompts["positive_prompt"] + "\n\n" + base_prompt + sample["answer_matching_behavior"],
                            "negative": dataset_prompts["positive_prompt"] + "\n\n" + base_prompt + sample["answer_not_matching_behavior"]
                        }
                    },
                    "negative_prompt": {
                        "prompt_text": dataset_prompts["negative_prompt"],
                        "prompts": {
                            "positive": dataset_prompts["negative_prompt"] + "\n\n" + base_prompt + sample["answer_matching_behavior"],
                            "negative": dataset_prompts["negative_prompt"] + "\n\n" + base_prompt + sample["answer_not_matching_behavior"]
                        }
                    }
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
            "scenarios": {}
        }
        
        # Process each scenario (base, positive_prompt, negative_prompt)
        for scenario_name, scenario_data in sample["scenarios"].items():
            sample_data["scenarios"][scenario_name] = {
                "prompt_text": scenario_data.get("prompt_text", ""),  # Empty for base scenario
                "prompts": scenario_data["prompts"],
                "logits": {"positive": {}, "negative": {}},
                "activations": {"positive": {}, "negative": {}}
            }
            
            # Process both positive and negative prompts for each scenario
            for sent_type in ["positive", "negative"]:
                inputs = tokenizer(scenario_data["prompts"][sent_type], return_tensors="pt").to(device)
                
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

def main() -> None:
    DATASETS_PATH = utils.get_path('DATASETS_PATH')
    SAMPLES_PATH = os.path.join(DATASETS_PATH, "reliability_paper")
    PROMPTS_PATH = os.path.join(SAMPLES_PATH, "prompts")
    NUM_SAMPLES = 3

    # Load prompts
    prompts = load_prompts(PROMPTS_PATH)
    datasets = [("interest-in-science", "persona"), ("survival-instinct", "x-risk")]
    datasets_dict = {name: os.path.join(SAMPLES_PATH, f"{type}/{name}.jsonl") for name, type in datasets}

    for dataset_name, file_path in datasets_dict.items():
        try:
            dataset_type = "persona"  # Since we know it's interest-in-science
            
            for batch in load_sentences(file_path, dataset_type, prompts, dataset_name, num_samples=NUM_SAMPLES):
                for sample in batch["samples"]:
                    print(f"\nQuestion: {sample['question']}\n")
                    
                    for scenario_name, scenario_data in sample["scenarios"].items():
                        print(f"\n{scenario_name.upper()}:")
                        for prompt_type, prompt in scenario_data["prompts"].items():
                            print(f"\n{prompt_type}:")
                            print("-" * 50)
                            print(prompt)
                            print("-" * 50)

        except ValueError as e:
            logging.error(f"Error processing dataset {dataset_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()