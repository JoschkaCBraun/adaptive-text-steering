'''
create_topic_vectors.py

This script creates a topic vectors. 

The steering vector depends on the following parameters:
- model_alias
- representation_type
- pairing_type
- tid
- num_samples
- language
'''

# Standard library imports
import logging
import pickle
from typing import Dict

# Third-party imports
from steering_vectors import train_steering_vector

from src.utils import load_model_and_tokenizer, get_topic_vector_file_path,
load_topic_vector_training_samples, save_topic_vector, load_topic_vector
from config.experiment_config import ExperimentConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_topic_vector(model_alias: str, layer: int, topic_representation_type: str, 
                     pairing_type: str, tid: int, num_samples: int, language: str = 'en'
                     ) -> Dict[str, float]:
    """
    Get or train the steering vector for a given topic id.
    
    :param model_alias: The alias of the model to use.
    :param layer: The layer of the model to use.
    :param topic_representation_type: The type of topic representation to use.
    :param pairing_type: The type of pairing to use.
    :param tid: The topic id for which to get or train the steering vector.
    :param language: The language of the topic.
    :param num_samples: The number of samples to use for training the steering vector.
    :return: The steering vector for the specified topic id.
    """
    topic_vector_file_path = get_topic_vector_file_path(model_alias, layer, topic_representation_type, pairing_type, tid, num_samples, language)

    try:
        with open(topic_vector_file_path, 'rb') as f:
            topic_vector = pickle.load(f)
        return topic_vector
    except FileNotFoundError:
        # If the file is not found, train all topic vectors for the specified encoding type
        logger.info(f"Topic vector for topic ID {tid} not found. Training all topic vectors.")
        train_topic_vectors(model_alias, layer, topic_representation_type, pairing_type, tid, num_samples, language)

        # Retry loading the newly trained vector
        try:
            with open(topic_vector_file_path, 'rb') as f:
                topic_vector = pickle.load(f)
            return topic_vector
        except FileNotFoundError:
            logger.error(f"Error loading topic vectors for topic ID {tid}, even after training.")
            return {}
        
def train_topic_vectors(model_alias: str, layer: int, topic_representation_type: str, pairing_type: str, tid: int, num_samples: int, language: str = 'en') -> None:
    ''' Train the steering vectors for each topic in the topic strings.

    :param topic_encoding_type: The type of topic encoding to use.
    '''

    tokenizer, model, _ = load_model_and_tokenizer(model_alias)

    training_samples = load_topic_vector_training_samples(topic_representation_type, pairing_type, tid, language)

    # select the first num_samples samples
    training_samples = training_samples[:num_samples]

    # train the steering vector
    steering_vector = train_steering_vector(model, tokenizer, training_samples, layers=[layer])

    # save the steering vector
    topic_vector_file_path = get_topic_vector_file_path(model_alias, layer, topic_representation_type, pairing_type, tid, num_samples, language)
    save_topic_vector(topic_vector_file_path, steering_vector)

# pylint: enable=logging-fstring-interpolation


if __name__ == "__main__":
    # get the topic vector for the topic id 1
    topic_vector = get_topic_vector(model_alias="llama3_1b", layer=12, topic_representation_type="topic_words", pairing_type="against_random_topic_representation", tid=12, num_samples=100, language="en")

    loaded_topic_vector = load_topic_vector(get_topic_vector_file_path(model_alias="llama3_1b", layer=12, topic_representation_type="topic_words", pairing_type="against_random_topic_representation", tid=12, num_samples=100, language="en"))