'''
get_steering_vectors.py

This script loads or trains steering vectors for different concepts and languages.
'''
import os
import sys
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd
from steering_vectors import train_steering_vector
from transformers import pipeline

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
def get_steering_vector(config: ExperimentConfig, concept: str, language: str, num_tuples: int,
                        layers: Optional[List[int]] = None, language2: Optional[str] = None,
                        conflicting: bool = False) -> Dict[str, float]:
    '''
    Get or train the steering vector for a given language, with options for specific layers
    and conflicting steering between two languages.
    
    :param config: ExperimentConfig object containing the configuration for the experiment
    :param concept: The concept for which to get the steering vector
    :param language: The primary language for which to get the steering vector
    :param num_tuples: The number of samples to use for training the steering vector
    :param layers: List of layers to train the steering vector on. If None, train on all layers.
    :param language2: The second language in which to train the steering vector.
    :param conflicting: Indicates whether steering of language1 and language2 conflict.
    :return: The steering vector for the specified language(s) and layers
    '''
    if concept not in ['sentiment', 'simplicity', 'toxicity']:
        logger.error(f"Concept {concept} not recognised.")
        return None

    file_path = get_file_path(config=config, file_type='steering_vector', concept=concept,
                              language=language, num_tuples=num_tuples, layers=layers,
                              language2=language2, conflicting=conflicting)
    steering_vector = load_steering_vector(file_path)

    if steering_vector is None:
        logger.info(f"Steering vector not found. Therefore, training it now.")
        steering_vector = train_steering_vectors(config, language, num_tuples, layers, language2, conflicting)
        if steering_vector:
            save_steering_vector(steering_vector, file_path)
            logger.info(f"Steering vector saved to {file_path}")
        else:
            logger.error(f"Failed to train steering vector")
    else:
        logger.info(f"Steering vector found.")
    return steering_vector

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
    if file_type == 'training_data':
        return os.path.join(folder_path, f'{concept}_vectors_training_data', file_name)
    elif file_type == 'steering_vector':
        return os.path.join(folder_path, f'{concept}_vectors_activations', file_name)
    else:
        logger.error(f"Type {file_type} not recognised.")

def get_file_name(config: ExperimentConfig, file_type: str, concept: str, language: str,
                  num_tuples: int, layers: Optional[List[int]] = None,
                  language2: Optional[str] = None, conflicting: bool = False) -> str:
    '''Get the file name for the file, incorporating layers, languages, and conflicts.'''
    training_dataset = get_dataset_name(concept)
    
    layer_str = "all" if layers is None else f"{layers[0]}_{layers[-1]}"
    conflict_str = f"_conflict" if conflicting else ""
    language_str = f"{language}_{language2}" if language2 else language
    
    if file_type == 'steering_vector':
        return f'{training_dataset}_vector_{language_str}_{num_tuples}_for_{config.MODEL_ALIAS}_{layer_str}{conflict_str}.pkl'
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

def load_steering_vector(file_path: str) -> Optional[Dict[str, float]]:
    """Load a steering vector from a file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.info(f"Steering vector not found at {file_path}")
        return None
    except IOError as e:
        logger.error(f"I/O error loading steering vector: {e}")
        return None
    except pickle.PickleError as e:
        logger.error(f"Error unpickling steering vector: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading steering vector: {e}")
        return None

def save_steering_vector(steering_vector: Dict[str, float], file_path: str) -> None:
    """Save a steering vector to a file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(steering_vector, f)
        logger.info(f"Steering vector saved to {file_path}")
    except IOError as e:
        logger.error(f"I/O error saving steering vector: {e}")
    except pickle.PickleError as e:
        logger.error(f"Error pickling steering vector: {e}")
    except Exception as e:
        logger.error(f"Error saving steering vector: {e}")

def train_steering_vectors(config: ExperimentConfig, concept: str, language: str, num_tuples: int,
                            layers: Optional[List[int]], language2: Optional[str] = None,
                            conflicting: bool = False) -> Optional[Dict[str, float]]:
    """
    Train the steering vectors for a given concept and language, potentially with conflicting
    steering.
    """
    tokenizer, model, _ = load_model_and_tokenizer(config=config, CustomModel=None)

    training_data = get_steering_training_data(config=config, concept=concept, language=language,
                                               num_tuples=num_tuples)
    training_tuples = [tuple(pair) for pair in training_data['tuples']]

    if language2:
        training_data2 = get_steering_training_data(config, concept, language2, num_tuples)
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

def translate_tuples(tuples: List[Tuple[str, str]], source_lang: str, target_lang: str
                     ) -> List[Tuple[str, str]]:
    '''
    Translate a list of tuples from a source language to a target language.

    :param tuples: A list of tuples to translate
    :param source_lang: The language of the source text
    :param target_lang: The language to translate the source text to.
    :return: A list of tuples with the source text translated to the target language.
    '''
    device = get_device(device_map='auto')
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    logger.info(f"Using model {model_name} with device {device} to translate tuples.")
    logger.info(f"Translating {len(tuples)} tuples from {source_lang} to {target_lang}")
    translation_pipeline = pipeline(task=f'translation_{source_lang}_to_{target_lang}',
                                    model=model_name,
                                    device=device)
    
    return [(translation_pipeline(tup[0])[0]['translation_text'],
             translation_pipeline(tup[1])[0]['translation_text']) for tup in tuples]

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
    LANGUAGES = ['en', 'de', 'fr']

    for language in LANGUAGES:
        steering_vector = get_steering_vector(config, language, NUM_TUPLES)
        if steering_vector:
            logger.info(f"Successfully processed steering vector for {language}")
        else:
            logger.error(f"Failed to process steering vector for {language}")

# pylint: enable=logging-fstring-interpolation
if __name__ == "__main__":
    main()
