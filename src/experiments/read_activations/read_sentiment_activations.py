import os
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations
import pickle

def load_sentences(file_path, n=100):
    sentences = {"positive": [], "negative": []}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= n:
                break
            sentences["positive"].append(row["positive"])
            sentences["negative"].append(row["negative"])
    return sentences

def record_model_activations(model, tokenizer, sentences, layer_nums, device):
    all_activations = {}
    
    for sentiment, sent_list in sentences.items():
        all_activations[sentiment] = []
        
        for sentence in sent_list:
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            
            with record_activations(model, layer_type="decoder_block", layer_nums=layer_nums) as recorded_activations:
                model(**inputs)
            
            all_activations[sentiment].append(recorded_activations)
    
    return all_activations

def save_activations(activations, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(activations, f)
    print(f"Activations saved to {file_path}")

def main():
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "google/gemma-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load sentences
    file_path = "data/datasets/sentiment/sentiment_synthetic_data/sentiment_sentences/sentiment_500_sentences_en.csv"
    sentences = load_sentences(file_path, n=10)

    # Set up layer numbers to record (every 3rd layer starting from the 2nd)
    num_layers = model.config.num_hidden_layers
    layer_nums = list(range(1, num_layers, 3))

    # Record activations
    activations = record_model_activations(model, tokenizer, sentences, layer_nums, device)

    # Save activations to file
    output_file = "data/read_activations/read_sentiment_activations/gemma_2b_activations.pkl"
    save_activations(activations, output_file)
if __name__ == "__main__":
    main()