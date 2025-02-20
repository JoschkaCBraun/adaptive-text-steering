import os
import sys
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from steering_vectors import train_steering_vector
from datasets import Dataset

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# pylint: disable=wrong-import-position
from src.utils.load_and_get_utils import load_model_and_tokenizer, load_dataset, get_data_dir, \
    get_device
from config.experiment_config import ExperimentConfig
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_simplicity_vector(config: ExperimentConfig, num_tuples: int,
                         layers: Optional[List[int]] = None) -> Dict[str, float]:
    '''
    Get or train the simplicity vector for English, with options for specific layers.
    
    :param config: ExperimentConfig object containing the configuration for the experiment
    :param num_tuples: The number of samples to use for training the simplicity vector
    :param layers: List of layers to train the simplicity vector on. If None, train on all layers.
    :return: The simplicity vector for the specified layers
    '''
    file_path = get_file_path(config=config, file_type='simplicity_vector',
                              num_tuples=num_tuples, layers=layers)
    simplicity_vector = load_simplicity_vector(file_path)
    
    if simplicity_vector is None:
        logger.info(f"Simplicity vector not found. Therefore, training it now.")
        simplicity_vector = train_simplicity_vectors(config, num_tuples, layers)
        if simplicity_vector:
            save_simplicity_vector(simplicity_vector, file_path)
            logger.info(f"Simplicity vector saved to {file_path}")
        else:
            logger.error(f"Failed to train simplicity vector")
    else:
        logger.info(f"Simplicity vector found.")
    return simplicity_vector
    
def get_file_path(config: ExperimentConfig, file_type: str, num_tuples: int,
                  layers: Optional[List[int]] = None) -> str:
    '''
    Get the path for the training data or simplicity vector file.

    :param config: ExperimentConfig object containing the configuration for the experiment
    :param file_type: The file type: 'training_data' or 'simplicity_vector'
    :param num_tuples: The number of samples to use for training the simplicity vector
    :param layers: List of layers to train the simplicity vector on. If None, train on all layers.
    :return: The file path for the specified file type
    '''
    folder_path = os.path.join(get_data_dir(os.getcwd()), 'simplicity_vectors_data')
    if file_type == 'training_data':
        return os.path.join(folder_path, 'simplicity_vectors_training_data',
                            get_file_name(config, file_type, num_tuples))
    elif file_type == 'simplicity_vector':
        return os.path.join(folder_path, 'simplicity_vectors_activations',
                            get_file_name(config, file_type, num_tuples, layers))
    else:
        logger.error(f"Type {file_type} not recognised.")

def get_file_name(config: ExperimentConfig, file_type: str, num_tuples: int,
                  layers: Optional[List[int]] = None) -> str:
    '''Get the file name for the file, incorporating layers if applicable.'''

    training_dataset = "simplicity_explanations"
    layer_str = "all" if layers is None else f"{layers[0]}_{layers[-1]}"
    
    if file_type == 'simplicity_vector':
        return f'{training_dataset}_topic_vector_en_{num_tuples}_for_{config.MODEL_ALIAS}_{layer_str}.pkl'
    if file_type == 'training_data':
        return f'{training_dataset}_{num_tuples}_en.json'
    else:
        logger.error(f"file_type {file_type} not recognised.")
        return None

def load_simplicity_vector(file_path: str) -> Optional[Dict[str, float]]:
    """Load a simplicity vector from a file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.info(f"Simplicity vector not found at {file_path}")
        return None
    except IOError as e:
        logger.error(f"I/O error loading simplicity vector: {e}")
        return None
    except pickle.PickleError as e:
        logger.error(f"Error unpickling simplicity vector: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading simplicity vector: {e}")
        return None

def save_simplicity_vector(simplicity_vector: Dict[str, float], file_path: str) -> None:
    """Save a simplicity vector to a file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(simplicity_vector, f)
        logger.info(f"Simplicity vector saved to {file_path}")
    except IOError as e:
        logger.error(f"I/O error saving simplicity vector: {e}")
    except pickle.PickleError as e:
        logger.error(f"Error pickling simplicity vector: {e}")
    except Exception as e:
        logger.error(f"Error saving simplicity vector: {e}")

def train_simplicity_vectors(config: ExperimentConfig, num_tuples: int,
                            layers: Optional[List[int]]) -> Optional[Dict[str, float]]:
    """Train the simplicity vectors."""
    tokenizer, model, _ = load_model_and_tokenizer(config=config, CustomModel=None)
    
    training_data = get_simplicity_training_data(config, num_tuples)
    if training_data is None:
        logger.error("Failed to get training data.")
        return None

    training_tuples = [tuple(pair) for pair in training_data['tuples']]

    return train_steering_vector(model=model, tokenizer=tokenizer, training_samples=training_tuples,
                                 layers=layers, show_progress=True)

def get_simplicity_training_data(config: ExperimentConfig, num_tuples: int) -> Optional[Dict]:
    """Get the training data from the JSON file."""
    file_path = get_file_path(config, 'training_data', num_tuples)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Ensure we only use the requested number of tuples
        data['tuples'] = data['tuples'][:num_tuples]
        return data
    else:
        logger.error(f"Training data not found at {file_path}")
        return None

def main():
    """Train simplicity vectors."""
    config = ExperimentConfig()
    NUM_TUPLES = 500
    
    simplicity_vector = get_simplicity_vector(config, NUM_TUPLES)
    if simplicity_vector:
        logger.info(f"Successfully processed simplicity vector")
    else:
        logger.error(f"Failed to process simplicity vector")

# pylint: enable=logging-fstring-interpolation
if __name__ == "__main__":
    main()