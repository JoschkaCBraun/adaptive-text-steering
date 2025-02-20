import os
import json
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import pickle
from typing import Dict, List, Generator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sentences(file_path: str, n: int = 50) -> Generator[Dict[str, List[str]], None, None]:
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)

    count = 0
    batch = []
    for sentence in data[:n]:
        batch.append(sentence)
        count += 1
        if len(batch) == 10:
            yield batch
            batch = []

    if batch:
        yield batch

    print(f"Loaded {count} sentences from {file_path}")

def record_model_activations(model, tokenizer, sentences, device, model_layer) -> Dict:
    first_activations = []
    last_activations = []
    average_activations = []

    for i, sentence in enumerate(sentences):
        logging.info(f"Processing sentence {i+1}/{len(sentences)}")
        inputs = tokenizer(sentence, return_tensors="pt").to(device)

        with record_activations(model, layer_type="decoder_block", layer_nums=model_layer) as recorded_activations:
            model(**inputs)
                
        sentence_first_activations = {}
        sentence_last_activations = {}
        sentence_average_activations = {}
        for layer, activations in recorded_activations.items():
            sentence_first_activations[layer] = activations[0][:, 1, :]
            sentence_last_activations[layer] = activations[0][:, -2, :]
            sentence_average_activations[layer] = torch.mean(activations[0], dim=1)
        
        first_activations.append(sentence_first_activations)
        last_activations.append(sentence_last_activations)
        average_activations.append(sentence_average_activations)
    
    return first_activations, last_activations, average_activations

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
    # disk path is /Volumes/WD_HDD_MAC/Masters_Thesis_data/
    DISK_PATH = "/Volumes/WD_HDD_MAC/Masters_Thesis_data/"
    DATASET_PATH = os.path.join(PROJECT_PATH, "data", "datasets")
    input_base_path = os.path.join(DATASET_PATH, "repe_datasets", "emotions")
    output_base_path = os.path.join(DISK_PATH, "repe_activations")

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    models = [
        ('gemma-2-2b-it', 'google/gemma-2-2b-it', [1, 5, 10, 13, 20, 25]), 
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [1, 5, 10, 13, 20, 25, 32]), # 13 indicates the layer
    ]

    num_samples = 100

    datasets = [
        "anger",
        "disgust",
        "fear",
        # "happiness_fear",
        # "happiness_sadness",
        "happiness",
        "sadness",
        "surprise",
    ]

    datasets = {name: os.path.join(input_base_path, f"{name}.json") for name in datasets}

    for model_name, model_path, model_layer in models:
        logging.info(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        num_layers = model.config.num_hidden_layers
        logging.info(f"Model {model_name} has {num_layers} layers")

        for dataset_name, file_path in datasets.items():
            logging.info(f"Processing dataset: {dataset_name}")
            
            first_file_name = f"{model_name}_first_activations_{dataset_name}_for_{num_samples}_samples.pkl"
            last_file_name = f"{model_name}_last_activations_{dataset_name}_for_{num_samples}_samples.pkl"
            average_file_name = f"{model_name}_average_activations_{dataset_name}_for_{num_samples}_samples.pkl"
            
            first_activations_file = os.path.join(output_base_path, first_file_name)
            last_activations_file = os.path.join(output_base_path, last_file_name)
            average_activations_file = os.path.join(output_base_path, average_file_name)
            
            for batch_num, sentences_batch in enumerate(load_sentences(file_path, n=num_samples)):
                first_activations, last_activations, average_activations = record_model_activations(model, tokenizer, sentences_batch, device, model_layer)

                mode = 'wb' if batch_num == 0 else 'ab'
                save_activations(first_activations, first_activations_file, dataset_name, mode)
                save_activations(last_activations, last_activations_file, dataset_name, mode)
                save_activations(average_activations, average_activations_file, dataset_name, mode)
                
                # Clear memory
                del first_activations, last_activations, average_activations

        del model, tokenizer

if __name__ == "__main__":
    main()