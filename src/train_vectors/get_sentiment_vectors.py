'''
get_sentiment_vectors.py

This script loads or trains sentiment vectors for different languages.
'''
import logging
import sys
import os
from typing import Dict, List, Optional

# Third-party imports
from steering_vectors import train_steering_vector

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import load_model_and_tokenizer, load_sentiment_vector_training_samples

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_sentiment_vector(model_alias: str, num_samples: int, representation_type: str,
                         language: str, layers: Optional[List[int]] = None) -> Dict[str, float]:
    '''
    Get or train the sentiment vector for a given language, with options for specific layers
    and conflicting sentiment between two languages.
    
    :param config: ExperimentConfig object containing the configuration for the experiment
    :param language: The primary language for which to get the sentiment vector
    :param num_tuples: The number of samples to use for training the sentiment vector
    :param layers: List of layers to train the sentiment vector on. If None, train on all layers.
    :return: The sentiment vector for the specified language(s) and layers
    '''
    tokenizer, model, _ = load_model_and_tokenizer(model_alias=model_alias)

    # Load training samples
    training_samples = load_sentiment_vector_training_samples(representation_type=representation_type, num_samples=num_samples, language=language)

    # Train sentiment vector
    steering_vector = train_steering_vector(model, tokenizer, training_samples, layers=layers,
                                            layer_type='decoder_block')

    return steering_vector
# pylint: enable=logging-fstring-interpolation


# test the function with main 
def main() -> None:
    model_alias = 'llama3_1b'
    num_samples = 100
    representation_type = 'sentiment_sentences'
    language = 'en'
    layers = [8]

    sentiment_vector = get_sentiment_vector(model_alias=model_alias, num_samples=num_samples, representation_type=representation_type, language=language, layers=layers)
    print(sentiment_vector)

if __name__ == "__main__":
    main()
