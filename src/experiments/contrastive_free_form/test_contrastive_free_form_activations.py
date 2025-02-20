import os
import sys
import pickle
from dotenv import load_dotenv
import torch
import logging
import time
import argparse
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Get the utils path from the environment variable
utils_path = os.getenv('UTILS_PATH')

# Add the utils path to sys.path if it's not None
if utils_path:
    sys.path.append(utils_path)
else:
    raise ValueError("UTILS_PATH environment variable is not set")

import compute_steering_vectors as sv

LayerActivations = Dict[int, List[torch.Tensor]]
SteeringVectors = Dict[int, torch.Tensor]

def load_activations(base_file_path: str, activation_type: str, load_num_samples: int,
                     compute_num_samples: int) -> Tuple[LayerActivations, LayerActivations]:
    """
    Load activations from a file, truncate to the specified number of samples, and print dimensionality.
    
    Args:
        base_file_path (str): The base file path for the activations.
        activation_type (str): The type of activations to load.
        load_num_samples (int): Number of samples to load.
    
    Returns:
        dict: A dictionary containing the loaded activations.
    
    Data Structure:
        The loaded data is a nested dictionary with the following structure:
        {
            'positive': {
                layer_number (int): [torch.Tensor(hidden_size), ...],
                ...
            },
            'negative': {
                layer_number (int): [torch.Tensor(hidden_size), ...],
                ...
            }
        }
        
        - The top-level keys are 'positive' and 'negative', representing activations for positive and negative samples.
        - The second-level keys are layer numbers (as integers) for each layer in the model.
        - The values are lists of torch.Tensor objects, each representing the activations for a layer and single sample.
        - Each tensor has shape (hidden_size,), where hidden_size is the dimensionality of the model's hidden state.
    """
    
    file_path = f"{base_file_path}_{activation_type}.pkl"
    logging.info(f"Loading {load_num_samples} samples of {activation_type} activations from {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    positive_activations = {}
    negative_activations = {}
    
    for key in ['positive', 'negative']:
        for layer in data[key]:
            if key == 'positive':
                positive_activations[layer] = data[key][layer][:compute_num_samples]
            else:
                negative_activations[layer] = data[key][layer][:compute_num_samples]
    
    # Log the number of samples loaded
    logging.info(f"Loaded {len(positive_activations[0])} samples of positive activations returned") 
    return positive_activations, negative_activations


def main(load_num_samples, compute_num_samples):
    start_time = time.time()
    logging.info(f"Starting main function with load_num_samples={load_num_samples}, compute_num_samples={compute_num_samples}")
    
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    if not PROJECT_PATH:
        raise ValueError("PROJECT_PATH not found in .env file")
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    DISK_PATH = os.getenv('DISK_PATH')
    if not DISK_PATH:
        raise ValueError("DISK_PATH not found in .env file")
    
    activations_path = os.path.join(DISK_PATH, "contrastive_free_form_activations")

    global device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    models = ['gemma-2-2b-it', 'qwen25_3b_it']# , 'llama32_3b_it', 'llama2_7b_chat']
    # models = ['qwen25_05b_it', 'qwen25_2b_it', 'qwen25_3b_it', 'qwen25_7b_it']
    datasets = [
        "truthfulness_306_en",
        # "sentiment_500_en",
        # "random_500",
        # "toxicity_500_en",
        # "active_to_passive_580_en",
    ]
    activation_types = ['penultimate', 'last', 'average']


    for model in models:
        for dataset in datasets:
            for activation_type in activation_types:
                base_file_path = os.path.join(activations_path, dataset, 
                                              f"{model}_activations_{dataset}_for_{load_num_samples}_samples")
                positive_activations, negative_activations = load_activations(base_file_path, 
                                                                              activation_type,
                                                                              load_num_samples,
                                                                              compute_num_samples)
                
                # Compute In-Context Vectors
                # log len(positive_activations), len(negative_activations)
                logging.info(f"Positive activations size: {len(positive_activations)}")
                # log keys of positive_activations
                logging.info(f"Positive activations keys: {positive_activations.keys()}")
                # log the type of values in positive_activations
                logging.info(f"Positive activations values type: {type(positive_activations[0])}")
                # log length of list of values in positive_activations
                logging.info(f"Positive activations values length: {len(positive_activations[0])}")
                in_context_vectors = sv.compute_in_context_vector(positive_activations, negative_activations, device)
                
                # Log shapes
                for layer, vector in in_context_vectors.items():
                    logging.info(f"Model: {model}, Dataset: {dataset}, Activation Type: {activation_type}, Layer: {layer}, ICV Shape: {vector.shape}")

    logging.info(f"Finished main function in {time.time() - start_time:.2f} seconds")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot activation distributions with sample size control.")
    parser.add_argument("--load_samples", type=int, default=50, help="Number of samples to load from each file")
    parser.add_argument("--compute_samples", type=int, default=50, help="Number of samples to use in computations")
    args = parser.parse_args()

    main(args.load_samples, args.compute_samples)



