import os
import json
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import pickle
from typing import Dict, List, Tuple, Generator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sentences(file_path: str, n: int = 50) -> Generator[Dict[str, List[str]], None, None]:
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for i in range(0, min(len(data), n), 10):
            batch = {"positive": [], "negative": []}
            for sample in data[i:i+10]:
                question = sample["question"]
                positive_answer = sample["answer_matching_behavior"]
                negative_answer = sample["answer_not_matching_behavior"]
                
                prompt_base = question + "\n\nYour response: "
                batch["positive"].append(prompt_base + f"{positive_answer}")
                batch["negative"].append(prompt_base + f"{negative_answer}")
            
            yield batch
    
    logging.info(f"Loaded {min(len(data), n)} sentence pairs from {file_path}")


def record_model_activations(model, tokenizer, sentences, device) -> Tuple[Dict, Dict]:
    penultimate_activations = {"positive": [], "negative": []}
    last_activations = {"positive": [], "negative": []}

    for sentiment, sent_list in sentences.items():
        for i, sentence in enumerate(sent_list):
            logging.info(f"Processing {sentiment} sentence {i+1}/{len(sent_list)}")
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            
            with record_activations(model, layer_type="decoder_block") as recorded_activations:
                model(**inputs)
            
            sentence_penultimate_activations = {}
            sentence_last_activations = {}
            for layer, activations in recorded_activations.items():
                # Only store every second layer
                if int(layer) % 2 == 0:
                    sentence_penultimate_activations[layer] = activations[0][:, -2, :]
                    sentence_last_activations[layer] = activations[0][:, -1, :]
            
            penultimate_activations[sentiment].append(sentence_penultimate_activations)
            last_activations[sentiment].append(sentence_last_activations)
    
    return penultimate_activations, last_activations

def save_activations(activations: Dict, file_path: str, dataset_name: str, mode: str = 'wb'):
    dataset_dir = os.path.join(os.path.dirname(file_path), dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    new_file_path = os.path.join(dataset_dir, os.path.basename(file_path))
    
    with open(new_file_path, mode) as f:
        pickle.dump(activations, f)
    logging.info(f"Activations {'appended to' if mode == 'ab' else 'saved to'} {new_file_path}")

def main():
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    input_base_path = os.path.join(DATA_PATH, "datasets", "caa_datasets", "generate")
    output_base_path = os.path.join(DATA_PATH, "caa_vectors_data", "caa_vectors_activations")

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    models = [
        # ("gemma-2-2b-it", "google/gemma-2-2b-it"),
        # ("gemma-2b-it", "google/gemma-2b-it"),
        ("llama-3-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct")
    ]

    num_samples = 100

    dataset_names = [
        "coordinate-other-ais",
        "hallucination",
        "refusal",
        "sycophancy",
        "corrigible-neutral-HHH",
        "myopic-reward",
        "survival-instinct"
    ]

    datasets = {name: os.path.join(input_base_path, f"{name}", "generate_dataset.json") for name in dataset_names}

    for model_name, model_path in models:
        logging.info(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        num_layers = model.config.num_hidden_layers
        logging.info(f"Model {model_name} has {num_layers} layers")

        for dataset_name, file_path in datasets.items():
            logging.info(f"Processing dataset: {dataset_name}")
            
            penultimate_file_name = f"{model_name}_penultimate_activations_{dataset_name}_{num_samples}.pkl"
            last_file_name = f"{model_name}_last_activations_{dataset_name}_{num_samples}.pkl"
            
            penultimate_activations_file = os.path.join(output_base_path, penultimate_file_name)
            last_activations_file = os.path.join(output_base_path, last_file_name)
            
            for batch_num, sentences_batch in enumerate(load_sentences(file_path, n=num_samples)):
                penultimate_activations, last_activations = record_model_activations(model, tokenizer, sentences_batch, device)
                
                mode = 'wb' if batch_num == 0 else 'ab'
                save_activations(penultimate_activations, penultimate_activations_file, dataset_name, mode)
                save_activations(last_activations, last_activations_file, dataset_name, mode)
                
                # Clear memory
                del penultimate_activations, last_activations

        del model, tokenizer

if __name__ == "__main__":
    main()