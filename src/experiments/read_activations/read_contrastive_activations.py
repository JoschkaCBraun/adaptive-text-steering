import os
import csv
import json
import torch
import pickle
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sentences(file_path, n=500):
    sentences = {"positive": [], "negative": []}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= n:
                break
            sentences["positive"].append(row["positive"])
            sentences["negative"].append(row["negative"])
    
    # Validation check
    if len(sentences["positive"]) != len(sentences["negative"]):
        raise ValueError("Number of positive and negative sentences do not match.")
    
    logging.info(f"Loaded {len(sentences['positive'])} pairs of sentences from {file_path}.")
    return sentences

def load_json_pairs(file_path):
    sentences = {"positive": [], "negative": []}
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for pair in data:
            sentences["positive"].append(pair[0])
            sentences["negative"].append(pair[1])
    
    logging.info(f"Loaded {len(sentences['positive'])} pairs of random sentences from {file_path}.")
    return sentences

def record_model_activations(model, tokenizer, sentences, device, batch_size=10):
    all_contrastive_activations = {}
    num_layers = model.config.num_hidden_layers
    
    for layer in range(num_layers):
        all_contrastive_activations[f"layer_{layer}"] = []
    
    total_pairs = len(sentences["positive"])
    
    for i in range(0, total_pairs, batch_size):
        batch_pos = sentences["positive"][i:i+batch_size]
        batch_neg = sentences["negative"][i:i+batch_size]
        
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_pairs + batch_size - 1)//batch_size}")

        pos_inputs = tokenizer(batch_pos, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        neg_inputs = tokenizer(batch_neg, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        try:
            with record_activations(model, layer_type="decoder_block") as pos_recorded_activations:
                model(**pos_inputs)
            
            with record_activations(model, layer_type="decoder_block") as neg_recorded_activations:
                model(**neg_inputs)
            
            for layer in range(num_layers):
                if layer not in pos_recorded_activations or layer not in neg_recorded_activations:
                    logging.warning(f"Layer {layer} not found in recorded activations. Skipping.")
                    continue
                
                pos_activation = pos_recorded_activations[layer]
                neg_activation = neg_recorded_activations[layer]
                
                if not isinstance(pos_activation, list) or not isinstance(neg_activation, list) or len(pos_activation) == 0 or len(neg_activation) == 0:
                    logging.warning(f"Unexpected activation format for layer {layer}. Skipping.")
                    continue
                
                pos_avg = calculate_average_activation(pos_activation[0])
                neg_avg = calculate_average_activation(neg_activation[0])
                contrastive = pos_avg - neg_avg
                all_contrastive_activations[f"layer_{layer}"].extend(contrastive)
        
        except Exception as e:
            logging.error(f"Error processing batch starting at sentence pair {i+1}: {str(e)}")
            raise
    
    logging.info(f"Processed activations for all {num_layers} layers and {total_pairs} sentence pairs.")
    return all_contrastive_activations

def calculate_average_activation(activation):
    if not isinstance(activation, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(activation)}")
    return activation.mean(dim=1).cpu().numpy()  # Average across tokens

def save_activations(activations, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(activations, f)
    logging.info(f"Activations saved to {file_path}")

def main():
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "google/gemma-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load sentences
    file_paths = {
        # "en": "data/datasets/sentiment/sentiment_synthetic_data/sentiment_sentences/sentiment_500_sentences_en.csv",
        # "de": "data/datasets/sentiment/sentiment_synthetic_data/sentiment_sentences/sentiment_500_sentences_de.csv",
        # "fr": "data/datasets/sentiment/sentiment_synthetic_data/sentiment_sentences/sentiment_500_sentences_fr.csv",
        "random": "data/datasets/random/random_pairs_500.json"
    }
    num_samples = 500  # Use all 500 samples
    
    try:
        for lang, path in file_paths.items():
            if lang == "random":
                sentences = load_json_pairs(path)
            else:
                sentences = load_sentences(path, n=num_samples)
            
            # Record activations
            contrastive_activations = record_model_activations(model, tokenizer, sentences, device)

            # Save activations
            output_file = f"data/read_activations/read_sentiment_activations/gemma_2b_all_layers_500_sentiment_activations_{lang}.pkl"
            save_activations(contrastive_activations, output_file)
        
        logging.info(f"Successfully processed and saved activations for {num_samples} samples across all layers.")
        
    except MemoryError:
        logging.error("Out of memory error occurred. Try reducing the batch size or using a machine with more RAM.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
