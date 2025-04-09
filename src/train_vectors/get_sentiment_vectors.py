'''
get_sentiment_vectors.py

This script loads or trains sentiment vectors for different languages.
'''
import os
import sys
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from steering_vectors import train_steering_vector
from datasets import Dataset

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# pylint: disable=wrong-import-position
from utils.load_topic_lda import load_model_and_tokenizer, load_dataset, get_data_dir, \
    get_device
from config.experiment_config import ExperimentConfig
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_sentiment_vector(config: ExperimentConfig, language: str, num_tuples: int,
                         layers: Optional[List[int]] = None, language2: Optional[str] = None,
                         conflicting: bool = False) -> Dict[str, float]:
    '''
    Get or train the sentiment vector for a given language, with options for specific layers
    and conflicting sentiment between two languages.
    
    :param config: ExperimentConfig object containing the configuration for the experiment
    :param language: The primary language for which to get the sentiment vector
    :param num_tuples: The number of samples to use for training the sentiment vector
    :param layers: List of layers to train the sentiment vector on. If None, train on all layers.
    :param language2: The second language in which to train the sentiment vector.
    :param conflicting: Indicates whether sentiment of language1 and language2 conflict.
    :return: The sentiment vector for the specified language(s) and layers
    '''
    file_path = get_file_path(config=config, file_type='sentiment_vector',
                              language=language, num_tuples=num_tuples,
                              layers=layers, language2=language2, conflicting=conflicting)
    sentiment_vector = load_sentiment_vector(file_path)
    
    if sentiment_vector is None:
        logger.info(f"Sentiment vector not found. Therefore, training it now.")
        sentiment_vector = train_sentiment_vectors(config, language, num_tuples, layers, language2, conflicting)
        if sentiment_vector:
            save_sentiment_vector(sentiment_vector, file_path)
            logger.info(f"Sentiment vector saved to {file_path}")
        else:
            logger.error(f"Failed to train sentiment vector")
    else:
        logger.info(f"Sentiment vector found.")
    return sentiment_vector
    
def get_file_path(config: ExperimentConfig, file_type: str, language: str, num_tuples: int) -> str:
    '''
    Get the path for the training data or sentiment vector file.

    :param config: ExperimentConfig object containing the configuration for the experiment
    :param file_type: The file type: 'training_data' or 'sentiment_vector'
    :param language: The language for which to get the file path
    :param num_tuples: The number of samples to use for training the sentiment vector
    :return: The file path for the specified file type
    '''
    folder_path = os.path.join(get_data_dir(os.getcwd()), 'sentiment_vectors_data')
    if file_type == 'training_data':
        return os.path.join(folder_path, 'sentiment_vectors_training_data',
                            get_file_name(config, file_type, language, num_tuples))
    elif file_type == 'sentiment_vector':
        return os.path.join(folder_path, 'sentiment_vectors_activations',
                            get_file_name(config, file_type, language, num_tuples))
    else:
        logger.error(f"Type {file_type} not recognised.")

def get_file_name(config: ExperimentConfig, file_type: str, language: str, num_tuples: int,
                  layers: Optional[List[int]] = None, language2: Optional[str] = None,
                  conflicting: bool = False) -> str:
    '''Get the file name for the file, incorporating layers, languages, and conflicts.'''

    training_dataset = "imdb_sentiment"
    layer_str = "all" if layers is None else f"{layers[0]}_{layers[-1]}"
    conflict_str = f"_conflict" if conflicting else ""
    language_str = f"{language}_{language2}" if language2 else language
    
    if file_type == 'sentiment_vector':
        return f'{training_dataset}_topic_vector_{language_str}_{num_tuples}_for_{config.MODEL_ALIAS}_{layer_str}{conflict_str}.pkl'
    if file_type == 'training_data':
        return f'{training_dataset}_train_{language}_{num_tuples}.json'
    else:
        logger.error(f"file_type {file_type} not recognised.")
        return None

def load_sentiment_vector(file_path: str) -> Optional[Dict[str, float]]:
    """Load a sentiment vector from a file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.info(f"Sentiment vector not found at {file_path}")
        return None
    except IOError as e:
        logger.error(f"I/O error loading sentiment vector: {e}")
        return None
    except pickle.PickleError as e:
        logger.error(f"Error unpickling sentiment vector: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading sentiment vector: {e}")
        return None

def save_sentiment_vector(sentiment_vector: Dict[str, float], file_path: str) -> None:
    """Save a sentiment vector to a file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(sentiment_vector, f)
        logger.info(f"Sentiment vector saved to {file_path}")
    except IOError as e:
        logger.error(f"I/O error saving sentiment vector: {e}")
    except pickle.PickleError as e:
        logger.error(f"Error pickling sentiment vector: {e}")
    except Exception as e:
        logger.error(f"Error saving sentiment vector: {e}")

def train_sentiment_vectors(config: ExperimentConfig, language: str, num_tuples: int,
                            layers: Optional[List[int]], language2: Optional[str] = None,
                            conflicting: bool = False) -> Optional[Dict[str, float]]:
    """Train the sentiment vectors for a given language, potentially with conflicting sentiment."""
    tokenizer, model, _ = load_model_and_tokenizer(config=config, CustomModel=None)
    
    training_data = get_sentiment_training_data(config, language, num_tuples)
    training_tuples = [tuple(pair) for pair in training_data['tuples']]

    if language2:
        training_data2 = get_sentiment_training_data(config, language2, num_tuples)
        if conflicting:
            training_tuples2 = [(neg, pos) for pos, neg in training_data2['tuples']]
        else:
            training_tuples2 = [tuple(pair) for pair in training_data2['tuples']]
        training_tuples.extend(training_tuples2)

    if training_tuples is None:
        logger.error("Failed to get training data.")
        return None

    return train_steering_vector(model=model, tokenizer=tokenizer, training_samples=training_tuples,
                                 layers=layers, show_progress=True)

# pylint: enable=logging-fstring-interpolation