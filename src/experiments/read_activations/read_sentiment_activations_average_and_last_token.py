import os
import csv
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import pickle
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

def load_sentences(file_path: str, n: int = 50) -> Dict[str, List[str]]:
    sentences = {"positive": [], "negative": []}
    
    if file_path.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                sentences["positive"].append(row["positive"])
                sentences["negative"].append(row["negative"])
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
            for i, pair in enumerate(data):
                if i >= n:
                    break
                sentences["positive"].append(pair[0])
                sentences["negative"].append(pair[1])
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logging.info(f"Loaded {len(sentences['positive'])} sentence pairs from {file_path}")
    return sentences

def record_model_activations(model, tokenizer, sentences, device) -> Tuple[Dict, Dict]:
    all_activations = {}
    last_activations = {}
    
    for sentiment, sent_list in sentences.items():
        all_activations[sentiment] = []
        last_activations[sentiment] = []
        
        for i, sentence in enumerate(sent_list):
            logging.info(f"Processing {sentiment} sentence {i+1}/{len(sent_list)}")
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            
            with record_activations(model, layer_type="decoder_block") as recorded_activations:
                model(**inputs)
            
            sentence_all_activations = {}
            sentence_last_activations = {}
            for layer, activations in recorded_activations.items():
                sentence_all_activations[layer] = torch.mean(activations[0], dim=1)
                sentence_last_activations[layer] = activations[0][:, -1, :]
            
            all_activations[sentiment].append(sentence_all_activations)
            last_activations[sentiment].append(sentence_last_activations)
    
    return all_activations, last_activations

def save_activations(activations: Dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(activations, f)
    logging.info(f"Activations saved to {file_path}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    models = [
        # ("gemma-2-2b-it", "google/gemma-2-2b-it"),
        # ("gemma-2b-it", "google/gemma-2b-it"),
        ("llama-3-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct")
    ]

    num_samples = 500

    sentiment_sentences_path = f"{PROJECT_PATH}/data/datasets/sentiment/sentiment_synthetic_data/sentiment_sentences"
    datasets = {
        # "en": f"{sentiment_sentences_path}/sentiment_500_sentences_en.csv",
        # "de": f"{sentiment_sentences_path}/sentiment_500_sentences_de.csv",
        "random": f"{PROJECT_PATH}/data/datasets/random/random_pairs_500.json"
    }

    for model_name, model_path in models:
        logging.info(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        num_layers = model.config.num_hidden_layers
        logging.info(f"Model {model_name} has {num_layers} layers")

        for dataset_name, file_path in datasets.items():
            logging.info(f"Processing dataset: {dataset_name}")
            sentences = load_sentences(file_path, n=num_samples)

            all_activations, last_activations = record_model_activations(model, tokenizer, sentences, device)

            sentiment_activations_path = f"{PROJECT_PATH}/data/read_activations/read_sentiment_activations/sentiment_activations"
            
            all_activations_file = f"{sentiment_activations_path}/{model_name}_all_activations_{dataset_name}_{num_samples}.pkl"
            last_activations_file = f"{sentiment_activations_path}/{model_name}_last_activations_{dataset_name}_{num_samples}.pkl"
            
            save_activations(all_activations, all_activations_file)
            save_activations(last_activations, last_activations_file)

        del model, tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()