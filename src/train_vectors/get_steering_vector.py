'''
get_steering_vectors.py

This script loads or trains steering vectors for different settings.
'''
import logging
import sys
import os
from typing import List, Optional

# Third-party imports
from steering_vectors import train_steering_vector, SteeringVector

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: disable=wrong-import-position
from src.utils import load_model_and_tokenizer, load_steering_vector, save_steering_vector
from src.create_training_samples.create_topic_vector_training_samples import create_topic_vector_training_samples
from src.create_training_samples.create_toxicity_vector_training_samples import create_toxicity_vector_training_samples
from src.create_training_samples.create_readability_vector_training_samples import create_readability_vector_training_samples
from src.create_training_samples.create_sentiment_vector_training_samples import create_sentiment_vector_training_samples
from config.experiment_config import ExperimentConfig
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_steering_vector(
        behavior_type: str,
        model_alias: str,
        num_samples: int,
        representation_type: str,
        steering_layers: List[int],
        language: str = 'en',
        pairing_type: Optional[str] = None,
        tid: Optional[int] = None) -> SteeringVector:
    '''
    Get or train the sentiment vector for a given language, with options for specific layers
    and conflicting sentiment between two languages.
    
    :param config: ExperimentConfig object containing the configuration for the experiment
    :param language: The primary language for which to get the sentiment vector
    :param num_tuples: The number of samples to use for training the sentiment vector
    :param layers: List of layers to train the sentiment vector on. If None, train on all layers.
    :return: The sentiment vector for the specified language(s) and layers
    '''
    config = ExperimentConfig()
    if behavior_type not in config.VALID_BEHAVIOR_TYPES:
        raise ValueError(f"Invalid behavior type: {behavior_type}. "
                         f"Expected one of {config.VALID_BEHAVIOR_TYPES}")
    
    if behavior_type == 'topic':
        # topic vectors are stored, so check if the file exists and load it
        topic_vector = load_steering_vector(behavior_type=behavior_type, model_alias=model_alias, representation_type=representation_type, num_samples=num_samples, steering_layers=steering_layers, language=language, pairing_type=pairing_type, tid=tid)
        if topic_vector is not None:
            return topic_vector

    tokenizer, model, _ = load_model_and_tokenizer(model_alias=model_alias)

    # Load training samples
    if behavior_type == 'sentiment':
        training_samples = create_sentiment_vector_training_samples(representation_type=representation_type, num_samples=num_samples, language=language)
    elif behavior_type == 'topic':
        training_samples = create_topic_vector_training_samples(topic_representation_type=representation_type, num_samples=num_samples, tid=tid, pairing_type=pairing_type)
    elif behavior_type == 'toxicity':
        training_samples = create_toxicity_vector_training_samples(num_samples=num_samples, representation_type=representation_type, language=language)
    elif behavior_type == 'readability':
        training_samples = create_readability_vector_training_samples(num_samples=num_samples, representation_type=representation_type, language=language)
    else:
        raise ValueError(f"Invalid behavior type: {behavior_type}. "
                         f"Expected one of {config.VALID_BEHAVIOR_TYPES}")

    print(f"Training steering vector for {behavior_type} with {len(training_samples)} samples")
    print(f"Training samples: {training_samples}")
    steering_vector = train_steering_vector(model=model, tokenizer=tokenizer, training_samples=training_samples, layers=steering_layers, layer_type='decoder_block')

    if behavior_type == 'topic':
        save_steering_vector(steering_vector=steering_vector, behavior_type=behavior_type, model_alias=model_alias, representation_type=representation_type, num_samples=num_samples, steering_layers=steering_layers, language=language, pairing_type=pairing_type, tid=tid)

    return steering_vector

# pylint: enable=logging-fstring-interpolation

# test the function with main 
def main() -> None:
    # for each behavior type, test the function with the llama3_1b model
    config = ExperimentConfig()
    sentiment_vector = get_steering_vector(behavior_type='sentiment', model_alias='llama3_1b', num_samples=10, representation_type='words', language='en', steering_layers=[8])

    print(f"Sentiment vector: {sentiment_vector}")

    topic_vector = get_steering_vector(behavior_type='topic', model_alias='llama3_1b', num_samples=10, representation_type='words', language='en', steering_layers=[8], pairing_type=config.PAIRING_TYPES[0], 
                                       tid=config.TOPIC_IDS[0])

    print(f"Topic vector: {topic_vector}")

    toxicity_vector = get_steering_vector(behavior_type='toxicity', model_alias='llama3_1b', num_samples=10, representation_type='words', language='en', steering_layers=[8])

    print(f"Toxicity vector: {toxicity_vector}")

    readability_vector = get_steering_vector(behavior_type='readability', model_alias='llama3_1b', num_samples=10, representation_type='words', language='en', steering_layers=[8])

    print(f"Readability vector: {readability_vector}")   

if __name__ == "__main__":
    main()
