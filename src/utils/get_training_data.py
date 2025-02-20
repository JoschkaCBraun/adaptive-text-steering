'''
get_training_data.py

This script retrieves or creates training data for steering vectors, linear probes and more.
Given a concept, language, dataset and number of tuples, it returns the requested training data.
'''
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# pylint: disable=wrong-import-position
from src.utils.load_and_get_utils import load_dataset, get_data_dir
from config.experiment_config import ExperimentConfig
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_file_path(config: ExperimentConfig, file_type: str, concept: str, language: str,
                  num_tuples: int, layers: Optional[List[int]] = None,
                  language2: Optional[str] = None, conflicting: bool = False) -> str:
    '''
    Get the path for the training data or steering vector file.

    :param config: ExperimentConfig object containing the configuration for the experiment
    :param file_type: The file type: 'training_data' or 'steering_vector'
    :param concept: The concept for which to get the file path
    :param language: The language for which to get the file path
    :param num_tuples: The number of samples to use for training the steering vector
    :param layers: List of layers to train the steering vector on. If None, retrieve steering vector trained on all layers.
    :param language2: The second language in which to train the steering vector.
    :param conflicting: Indicates whether steering of language1 and language2 conflict.
    :return: The file path for the specified file type
    '''
    folder_path = os.path.join(get_data_dir(os.getcwd()), f'{concept}_vectors_data')
    file_name = get_file_name(config, file_type, concept, language, num_tuples, layers, language2,
                              conflicting)
    if file_type == 'steering_vector':
        return os.path.join(folder_path, f'{concept}_vectors_activations', file_name)
    else:
        logger.error(f"Type {file_type} not recognised.")

def get_file_name(config: ExperimentConfig, file_type: str, concept: str, language: str,
                  num_tuples: int, layers: Optional[List[int]] = None,
                  language2: Optional[str] = None, conflicting: bool = False) -> str:
    '''Get the file name for the file, incorporating layers, languages, and conflicts.'''
    training_dataset = get_dataset_name(concept)

    if file_type == 'training_data':
        return f'{training_dataset}_train_{language}_{num_tuples}.json'
    else:
        logger.error(f"file_type {file_type} not recognised.")
        return None
    
def get_dataset_name(concept: str) -> str:
    '''Get the name of the dataset for a given concept.'''
    if concept == 'sentiment':
        return 'imdb_sentiment_train'
    elif concept == 'simplicity':
        return 'wiki_doc'
    elif concept == 'toxicity':
        return 'jigsaw_toxicity'
    else:
        logger.error(f"Concept {concept} not recognised.")
        return None

def get_steering_training_data(config: ExperimentConfig, concept: str, language: str,
                               num_tuples: int) -> Optional[Dict]:
    """Get the training data for a given concept and language."""
    file_path = get_file_path(config=config, file_type='training_data', concept=concept,
                              language=language, num_tuples=num_tuples)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.info("Training data not found. Generating it now.")
        return generate_and_store_training_data(config, concept, language, num_tuples,
                                                dataset_name=get_dataset_name(concept))

def create_training_data(data: pd.DataFrame, num_tuples: int, max_length: Optional[int] = 200
                         ) -> List[Tuple[str, str]]:
    '''
    Create tuples of positive and negative samples from the dataset. Only use samples with 
    a token count of less than or equal to the max_length.

    :param data: The dataset to create the training data from
    :param num_tuples: The number of tuples to create
    :param max_length: The maximum token count for a sample. Introduced for efficiency.
    :return: A list of tuples of positive and negative samples
    '''
    data['token_count'] = data['text'].apply(lambda x: len(x.split()))
    data_short = data[data['token_count'] <= max_length]

    positive_samples = data_short[data_short['label'] == 1].head(num_tuples)
    negative_samples = data_short[data_short['label'] == 0].head(num_tuples)

    if len(positive_samples) < num_tuples or len(negative_samples) < num_tuples:
        logger.error(f"Insufficient samples for training data. "
                     f"Positive samples: {len(positive_samples)}, "
                     f"Negative samples: {len(negative_samples)}. "
                     "Dataset might be too small or max_length too low.")
        return None

    return list(zip(positive_samples['text'], negative_samples['text']))

def generate_and_store_training_data(config: ExperimentConfig, concept= str, languages: List[str],
                                     num_tuples: int, dataset_name: str) -> Optional[Dict]:
    """Generate and store training tuples for each language."""

    data = load_dataset(dataset_name)
    training_tuples_en = create_training_data(data, num_tuples)
    
    for lang in languages:
        if lang == 'en':
            training_tuples = training_tuples_en
        else:
            training_tuples = translate_tuples(training_tuples_en, source_lang='en',
                                               target_lang=lang)
        
        data = {
            'sample_size': num_tuples,
            'language': lang,
            'dataset': dataset_name,
            'tuples': training_tuples
        }
        
        file_path = get_file_path(config, 'training_data', lang, num_tuples)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"Training data for {lang} saved to {file_path}")
    
    return data if languages else None

def main():
    """Train steering vectors for specified languages."""
    config = ExperimentConfig()
    NUM_TUPLES = 5
    LANGUAGES = ['en', 'de', 'fr', 'ru', 'es', 'it', 'zh']
    CONCEPTS = ['sentiment', 'simplicity', 'toxicity']

# pylint: enable=logging-fstring-interpolation
if __name__ == "__main__":
    main()
