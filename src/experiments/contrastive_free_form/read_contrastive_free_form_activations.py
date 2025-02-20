'''
read_contrastive_free_form_activations.py
'''
import os
import csv
import pickle
import logging
from typing import Dict, List, Generator, Tuple, Any, Literal
from steering_vectors import record_activations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import src.utils._validate_custom_datatypes as vcd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ActivationType = Literal['penultimate', 'last', 'average']
SentimentType = Literal['positive', 'negative']
class ExperimentConfig:
    def __init__(self, batch_size: int, num_samples: int, device: torch.device):
        """
        Initializes the configuration for model processing.

        Args:
            batch_size (int): Number of sentence pairs to process in a batch.
            num_samples (int): Number of samples to process.
            device (torch.device): The device to run the model on.
        """
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device

    def __repr__(self):
        return f"ModelConfig(batch_size={self.batch_size}, num_samples={self.num_samples}, device={self.device})"

def load_sentence_pairs(file_path: str, config: ExperimentConfig
                        ) -> Generator[List[Tuple[str, str]], None, None]:
    """
    Load sentence pairs from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        config (ExperimentConfig): Configuration for the experiment.
    
    Yields: 
        Generator[List[Tuple[str, str]], None, None]: A generator that yields a list of sentence pairs.
    """
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)[:config.num_samples]

    batch = []
    for row in data:
        batch.append((row['positive'], row['negative']))
        if len(batch) == config.batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

    print(f"Loaded {len(data)} sentence pairs from {file_path}")

def record_model_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence_pairs: List[Tuple[str, str]],
    model_layer: List[int],
    config : ExperimentConfig
) -> Dict[str, Dict[str, Dict[int, List[torch.Tensor]]]]:
    """
    Record model activations for given sentence pairs.

    Args:
        model (AutoModelForCausalLM): The language model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        sentence_pairs (List[Tuple[str, str]]): List of positive and negative sentence pairs.
        model_layer (List[int]): List of layer numbers to record activations from.
        config (ExperimentConfig): Configuration for the experiment.

    Returns:
        Dict[str, Dict[str, Dict[int, List[torch.Tensor]]]]: Recorded activations for positive and negative sentences.
    """
    activations = {
        'penultimate': {'positive': {}, 'negative': {}},
        'last': {'positive': {}, 'negative': {}},
        'average': {'positive': {}, 'negative': {}}
    }

    for positive, negative in sentence_pairs:
        for sentence, sent_type in [(positive, 'positive'), (negative, 'negative')]:
            inputs = tokenizer(sentence, return_tensors="pt").to(config.device)

            with record_activations(model, layer_type="decoder_block", layer_nums=model_layer) as recorded_activations:
                model(**inputs)
                
            for layer, layer_activations in recorded_activations.items():
                if layer not in activations['penultimate'][sent_type]:
                    activations['penultimate'][sent_type][layer] = []
                    activations['last'][sent_type][layer] = []
                    activations['average'][sent_type][layer] = []

                # layer_activations is a list of length 1
                layer_activations = layer_activations[0]

                # layer_activations is now a torch tensor of shape (1, num_tokens, hidden_size)
                layer_activations = layer_activations.squeeze(0)
                
                # layer_activations is now a torch tensor of shape (num_tokens, hidden_size)
                activations['penultimate'][sent_type][layer].append(layer_activations[-2, :])
                activations['last'][sent_type][layer].append(layer_activations[-1, :])
                activations['average'][sent_type][layer].append(torch.mean(layer_activations, dim=0))
            
    return activations

def save_activations(activations: Dict[str, Any], base_file_path: str, dataset_name: str, mode: str = 'wb') -> None:
    """
    Save activations to separate files for penultimate, last, and average.

    Args:
        activations (Dict[str, Any]): The activations to save.
        base_file_path (str): The base file path to save to.
        dataset_name (str): The name of the dataset.
        mode (str): The mode to open the file in ('wb' for write, 'ab' for append').
    """
    dataset_dir = os.path.join(os.path.dirname(base_file_path), dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    for activation_type in ['penultimate', 'last', 'average']:
        vcd.validate_activation_list_dict_to_activation_list_dict_compatibility(
            activations1=activations[activation_type]["positive"], 
            activations2=activations[activation_type]["negative"]
            )
        file_name = f"{os.path.basename(base_file_path)}_{activation_type}.pkl"
        file_path = os.path.join(dataset_dir, file_name)
        
        with open(file_path, mode) as f:
            pickle.dump(activations[activation_type], f)
        logging.info(f"{activation_type.capitalize()} activations {'appended to' if mode == 'ab' else 'saved to'} {file_path}")

def accumulate_activations(
    current_activations: Dict[str, Any],
    new_activations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Accumulate new activations into the current activations dictionary.

    Args:
        current_activations (Dict[str, Any]): The current activations to accumulate into.
        new_activations (Dict[str, Any]): The new activations to add.

    Returns:
        Dict[str, Any]: Updated activations with new data accumulated.
    """
    for act_type in current_activations.keys():
        for sent_type in ['positive', 'negative']:
            for layer in new_activations[act_type][sent_type]:
                if layer not in current_activations[act_type][sent_type]:
                    current_activations[act_type][sent_type][layer] = []
                current_activations[act_type][sent_type][layer].extend(new_activations[act_type][sent_type][layer])
    return current_activations

def process_model(model_name: str, model_path: str, model_layers: List[int], activations_path: str,
                  datasets: Dict[str, str], config: ExperimentConfig, ) -> None:
    """
    Process a single model for all datasets.

    Args:
        model_name (str): The name of the model.
        model_path (str): The path to load the model from.
        model_layer (List[int]): List of layer numbers to record activations from.
        activations_path (str): The path to save activations to.
        datasets (Dict[str, str]): Dictionary of dataset names and their file paths.
        config (ExperimentConfig): Configuration for the experiment.
    """
    logging.info(f"Processing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(config.device)
    
    num_layers = model.config.num_hidden_layers
    logging.info(f"Model {model_name} has {num_layers} layers")

    for dataset_name, file_path in datasets.items():
        logging.info(f"Processing dataset: {dataset_name}")
        
        base_file_name = f"{model_name}_activations_{dataset_name}_for_{config.num_samples}_samples"
        base_file_path = os.path.join(activations_path, base_file_name)
        
        accumulated_activations = {
            'penultimate': {'positive': {}, 'negative': {}},
            'last': {'positive': {}, 'negative': {}},
            'average': {'positive': {}, 'negative': {}}
        }

        for _, sentence_pairs in enumerate(load_sentence_pairs(file_path, config)):
            activations = record_model_activations(model, tokenizer, sentence_pairs, model_layers, config)

            # Accumulate activations
            accumulated_activations = accumulate_activations(accumulated_activations, activations)
            del activations

        # Save after all batches are processed
        save_activations(accumulated_activations, base_file_path, dataset_name, mode='wb')

    del model, tokenizer

def main():
    BATCH_SIZE = 10
    NUM_SAMPLES = 500
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    config = ExperimentConfig(batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES, device=device)

    DATASETS_PATH = os.getenv('DATASETS_PATH')
    DISK_PATH = os.getenv('DISK_PATH')
    if not DATASETS_PATH:
        raise ValueError("DATASETS_PATH not found in .env file")
    if not DISK_PATH:
        raise ValueError("DISK_PATH not found in .env file")
    
    DATASET_PATH = os.path.join(DATASETS_PATH, "datasets_ordered_by_type", "contrastive_free_form_datasets")
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "contrastive_free_form_activations")

    models = [
        ('gemma-2-2b-it', 'google/gemma-2-2b-it', [0, 5, 10, 15, 20, 25]), # 26 layers 
        # ('qwen25_3b_it', 'Qwen/Qwen2.5-3B-Instruct', [0, 5, 10, 15, 20, 25, 30, 35]), # 36 layers
        # ('llama32_3b_it', 'meta-llama/Llama-3.2-3B-Instruct', [0, 5, 10, 15, 20, 25, 27]), # 28 layers
        # ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [0, 5, 10, 15, 20, 25, 31]), # 32 layers
        # ('qwen25_05b_it', 'Qwen/Qwen2.5-0.5B-Instruct', [0, 5, 10, 15, 20, 23]) # 24 layers
        # ('qwen25_2b_it', 'Qwen/Qwen2.5-1.5B-Instruct', [0, 5, 10, 15, 20, 25, 27]) # 28 layers
        # ('qwen25_7b_it', 'Qwen/Qwen2.5-7B-Instruct', [0, 5, 10, 15, 20, 25, 27]) # 28 layers        
    ]

    datasets = [
        # "sentiment_500_en",
        # "random_500",
        # "toxicity_500_en",
        # "active_to_passive_500_en",
        # "truthfulness_500_en",
        # "capitalization_500_en",
        # "present_to_past_500_en",
        # "sentiment_500_fr",
        # "simplicity_500_en",
        # "translation_english_to_german_500",
    ]

    datasets = {name: os.path.join(DATASET_PATH, f"{name}.csv") for name in datasets}

    for model_name, model_path, model_layers in models:
        process_model(model_name, model_path, model_layers, ACTIVATIONS_PATH, datasets, config)

if __name__ == "__main__":
    main()
