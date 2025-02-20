'''
read_activations_for_reliability_paper_data.py
'''
import os
import json
import logging
import pickle
from typing import Dict, List, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import src.utils as utils
from src.utils.dataset_names import figure_13_reliability_paper_all_datasets_with_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sentences(file_path: str, dataset_type: str, num_samples: int = 10, batch_size: int = 10
                   ) -> Generator[Dict[str, List[str]], None, None]:
    """
    Load sentences from a file and yield them in batches.
    """
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        count = 0
        batch: Dict[str, list] = {"positive": [], "negative": []}
        for line in jsonfile:
            if count >= num_samples:
                break
            sample = json.loads(line)
            question = sample["question"]
            positive_answer = sample["answer_matching_behavior"]
            negative_answer = sample["answer_not_matching_behavior"]

            if dataset_type == "x-risk":
                prompt_base = f"Question: {question}\n\nAnswer:"
                positive_answer = positive_answer[:-1]
                negative_answer = negative_answer[:-1]
            elif dataset_type == "persona":
                prompt_base = f"Question: {question}\n\nChoices:\n - Yes\n - No\n\nAnswer:"
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            batch["positive"].append(prompt_base + positive_answer)
            batch["negative"].append(prompt_base + negative_answer)

            # print(f"Positive Response:\n{prompt_base + positive_answer}")
            # print(f"Negative Response:\n{prompt_base + negative_answer}")
            count += 1
            
            if len(batch["positive"]) == batch_size:
                yield batch
                batch = {"positive": [], "negative": []}
        
        # Yield any remaining examples
        if batch["positive"]:
            yield batch
    
    print(f"Loaded {count} sentence pairs from {file_path}")

def record_model_activations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                             sentences: Dict[str, List[str]], device: torch.device,
                             model_layer: List[int]) -> Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]]:
    """
    Record activations and compare penultimate activations between positive/negative pairs.
    """
    activations: Dict[str, Dict[int, list]] = {'positive': {layer: [] for layer in model_layer},
                                               'negative': {layer: [] for layer in model_layer}}

    for i, (pos_sent, neg_sent) in enumerate(zip(sentences['positive'], sentences['negative'])):
        # Process positive example
        pos_inputs = tokenizer(pos_sent, return_tensors="pt").to(device)
        with record_activations(model, layer_type="decoder_block", layer_nums=model_layer) as pos_activations:
            model(**pos_inputs)
            
        # Process negative example
        neg_inputs = tokenizer(neg_sent, return_tensors="pt").to(device)
        with record_activations(model, layer_type="decoder_block", layer_nums=model_layer) as neg_activations:
            model(**neg_inputs)
        
        # Compare penultimate activations and store last activations
        for layer in model_layer:
            pos_act = pos_activations[layer][0].squeeze(0)
            neg_act = neg_activations[layer][0].squeeze(0)
            
            # Store last token activations
            activations['positive'][layer].append(pos_act[-1, :].detach().cpu())
            activations['negative'][layer].append(neg_act[-1, :].detach().cpu())
            
            # Compare penultimate activations between positive and negative
            if len(pos_act) >= 2 and len(neg_act) >= 2:
                pos_penult = pos_act[-2, :].detach().cpu()
                neg_penult = neg_act[-2, :].detach().cpu()
                penult_diff = torch.abs(pos_penult - neg_penult).mean().item()
                if penult_diff > 1e-6:
                    logging.warning(f"Example {i}, Layer {layer}: Positive vs Negative penultimate activations are NOT identical (diff: {penult_diff:.2e})")
    
    return activations

def save_activations(activations: Dict, file_path: str, dataset_name: str) -> None:
    dataset_dir = os.path.join(os.path.dirname(file_path), dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    processed_activations = {
        sentiment: {
            layer: torch.stack(tensors) 
            for layer, tensors in activations[sentiment].items()
        }
        for sentiment in ['positive', 'negative']
    }
    
    out_path = os.path.join(dataset_dir, f"{os.path.basename(file_path)}_last.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(processed_activations, f)
    logging.info(f"Saved last token activations to {out_path}")

def main() -> None:
    DATASETS_PATH = utils.get_path('DATASETS_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    
    SAMPLES_PATH = os.path.join(DATASETS_PATH, "reliability_paper")
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    NUM_SAMPLES = 500

    device = utils.get_device()
    logging.info(f"Using device: {device}")

    models = [
        # ('gemma-2-2b-it', 'google/gemma-2-2b-it', [0, 5, 10, 13, 20, 25]),
        # ('llama-3-3b-it', 'meta-llama/Llama-3.2-3B-Instruct', [13]),
        # ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [1, 5, 10, 13, 20]), # 13 indicates the layer
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [13]), # 13 indicates the layer
        # ('qwen1.5_14b_chat', 'Qwen/Qwen1.5-14B-Chat', [1, 10, 21, 30, 40]) # 21 indicates the layer
    ]


    datasets = figure_13_reliability_paper_all_datasets_with_type

    x_risk_datasets = {dataset_name for dataset_name, dataset_type in datasets if dataset_type == "xrisk"}
    persona_datasets = {dataset_name for dataset_name, dataset_type in datasets if dataset_type == "persona"}

    # if dataset in x_risk_datasets add folder xrisk, if dataset in persona_datasets add folder persona
    datasets_dict = {name: os.path.join(SAMPLES_PATH, f"{type}/{name}.jsonl") for name, type in datasets}

    for model_name, model_path, model_layer in models:
        logging.info(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        for dataset_name, file_path in datasets_dict.items():
            logging.info(f"Processing dataset: {dataset_name}")
            accumulated_activations: Dict[Dict[str, list]]= {
                'positive': {layer: [] for layer in model_layer},
                'negative': {layer: [] for layer in model_layer}}

            if dataset_name in x_risk_datasets:
                dataset_type = "x-risk"
            elif dataset_name in persona_datasets:
                dataset_type = "persona"
            else:
                raise ValueError(f"Unknown dataset type for: {dataset_name}")

            for sentences_batch in load_sentences(file_path, dataset_type, num_samples=NUM_SAMPLES):
                batch_activations = record_model_activations(model, tokenizer, sentences_batch, device, model_layer)
                for sentiment in ['positive', 'negative']:
                    for layer in model_layer:
                        accumulated_activations[sentiment][layer].extend(
                            batch_activations[sentiment][layer]
                        )
            
            base_file_name = f"{model_name}_activations_{dataset_name}_for_{NUM_SAMPLES}_samples"
            base_file_path = os.path.join(ACTIVATIONS_PATH, base_file_name)
            save_activations(accumulated_activations, base_file_path, dataset_name)
            del accumulated_activations

        del model, tokenizer

if __name__ == "__main__":
    main()