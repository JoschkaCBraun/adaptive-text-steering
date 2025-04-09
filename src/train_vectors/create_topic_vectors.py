'''
create_topic_vectors.py

This script creates a topic vector for each topic.
'''
# Standard library imports
import os
import sys
import logging
import pickle
from typing import Dict, Optional, List, Tuple

# Third-party imports
import pandas as pd
from steering_vectors import train_steering_vector

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
# pylint: disable=wrong-import-position
from src.utils import load_model_and_tokenizer, load_dataset, load_lda, get_topic_words, get_data_dir
from src.utils import save_topic_training_samples, load_topic_training_samples
from config.experiment_config import ExperimentConfig
from src.utils.training_samples_utils import load_training_samples
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_topic_vector(config: ExperimentConfig, tid: int, topic_encoding_type: str,
                     data_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Get or train the steering vector for a given topic id.
    
    :param config: The Config object containing the configuration parameters.
    :param tid: The topic id for which to get or train the steering vector.
    :param topic_encoding_type: The type of topic encoding to use.
    :return: The steering vector for the specified topic id.
    """
    if data_dir:
        data_path = data_dir
    else:
        data_path = get_data_dir(os.getcwd())
    topic_vector_folder_name = get_topic_vectors_folder_name(topic_encoding_type, config)
    topic_vectors_path = os.path.join(data_path, 'topic_vectors_data', topic_vector_folder_name)
    results_dir = os.path.join(topic_vectors_path, f'topic_vector_tid_{tid}.pkl')

    try:
        with open(results_dir, 'rb') as f:
            topic_vector = pickle.load(f)
        return topic_vector
    except FileNotFoundError:
        # If the file is not found, train all topic vectors for the specified encoding type
        logger.info(f"Topic vector for topic ID {tid} not found. Training all topic vectors.")
        train_topic_vectors(config=config, topic_encoding_type=topic_encoding_type)

        # Retry loading the newly trained vector
        try:
            with open(results_dir, 'rb') as f:
                topic_vector = pickle.load(f)
            return topic_vector
        except FileNotFoundError:
            logger.error(f"Error loading topic vectors for topic ID {tid}, even after training.")
            return None

def get_topic_vectors_folder_name(topic_encoding_type: str, config: ExperimentConfig) -> str:
    '''Get the folder name for storing topic vectors.'''
    config_dict = config.TOPIC_VECTORS_CONFIG[topic_encoding_type]
    model_alias = config.MODEL_ALIAS
    folder_name = f"topic_vectors_from_{config_dict['topic_encoding_type']}_for_{model_alias}"

    if config_dict['topic_encoding_type'] == 'topical_summaries':
        if config_dict['include_non_matching']:
            folder_name += f"_fill_non_matching_up_to_{config_dict['num_samples']}_samples"
        else:
            folder_name += '_only_matching'
    elif config_dict['topic_encoding_type'] == 'zeros':
        folder_name = f'topic_vectors_all_zeros_for_{model_alias}'
    return folder_name

def train_topic_vectors(config: ExperimentConfig, topic_encoding_type: str) -> None:
    ''' Train the steering vectors for each topic in the topic strings.

    :param topic_encoding_type: The type of topic encoding to use.
    '''

    tokenizer, model, _ = load_model_and_tokenizer(config=config, CustomModel=None)

    training_samples = create_training_samples(config=config,
                                               topic_encoding_type=topic_encoding_type)
    topic_counter = 0
    num_topics = len(training_samples)

    folder_name = get_topic_vectors_folder_name(config=config,
                                                topic_encoding_type=topic_encoding_type)
    results_dir = os.path.join(parent_dir, 'data', 'topic_vectors_data', folder_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Train steering vectors for each topic
    for tid, training_samples in training_samples.items():
        topic_vector = train_steering_vector(model=model,
                                           tokenizer=tokenizer,
                                           training_samples=training_samples,
                                           show_progress=True)
        topic_vector_dir = os.path.join(results_dir, f'topic_vector_tid_{str(tid)}.pkl')
        with open(topic_vector_dir, 'wb') as f:
            pickle.dump(topic_vector, f)
        # log how many topics have been processed
        topic_counter += 1
        logger.info(f"Saved topic vector for topic id {tid} to {topic_vector_dir}"\
                    f"({topic_counter}/{num_topics})")

# pylint: enable=logging-fstring-interpolation
