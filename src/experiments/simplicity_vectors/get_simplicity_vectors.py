'''
get_simplicity_vectors.py

This script loads or trains simplicity vectors for different languages.
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
from src.utils.load_and_get_utils import load_model_and_tokenizer, load_dataset, get_data_dir, \
    get_device
from config.experiment_config import ExperimentConfig
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def get_simplicity_vector(config: ExperimentConfig, language: str, num_tuples: int,
                         layers: Optional[List[int]] = None, language2: Optional[str] = None,
                         conflicting: bool = False) -> Dict[str, float]:
    '''
    Get or train the simplicity vector for a given language, with options for specific layers
    and conflicting simplicity between two languages.
    
    :param config: ExperimentConfig object containing the configuration for the experiment
    :param language: The primary language for which to get the simplicity vector
    :param num_tuples: The number of samples to use for training the simplicity vector
    :param layers: List of layers to train the simplicity vector on. If None, train on all layers.
    :param language2: The second language in which to train the simplicity vector.
    :param conflicting: Indicates whether simplicity of language1 and language2 conflict.
    :return: The simplicity vector for the specified language(s) and layers
    '''
    file_path = get_file_path(config=config, file_type='simplicity_vector',
                              language=language, num_tuples=num_tuples,
                              layers=layers, language2=language2, conflicting=conflicting)
    simplicity_vector = load_simplicity_vector(file_path)
    
    if simplicity_vector is None:
        logger.info(f"Simplicity vector not found. Therefore, training it now.")
        simplicity_vector = train_simplicity_vectors(config, language, num_tuples, layers, language2, conflicting)
        if simplicity_vector:
            save_simplicity_vector(simplicity_vector, file_path)
            logger.info(f"Simplicity vector saved to {file_path}")
        else:
            logger.error(f"Failed to train simplicity vector")
    else:
        logger.info(f"Simplicity vector found.")
    return simplicity_vector
    
def get_file_path(config: ExperimentConfig, file_type: str, language: str, num_tuples: int) -> str:
    '''
    Get the path for the training data or simplicity vector file.

    :param config: ExperimentConfig object containing the configuration for the experiment
    :param file_type: The file type: 'training_data' or 'simplicity_vector'
    :param language: The language for which to get the file path
    :param num_tuples: The number of samples to use for training the simplicity vector
    :return: The file path for the specified file type
    '''
    folder_path = os.path.join(get_data_dir(os.getcwd()), 'simplicity_vectors_data')
    if file_type == 'training_data':
        return os.path.join(folder_path, 'simplicity_vectors_training_data',
                            get_file_name(config, file_type, language, num_tuples))
    elif file_type == 'simplicity_vector':
        return os.path.join(folder_path, 'simplicity_vectors_activations',
                            get_file_name(config, file_type, language, num_tuples))
    else:
        logger.error(f"Type {file_type} not recognised.")

def get_file_name(config: ExperimentConfig, file_type: str, language: str, num_tuples: int,
                  layers: Optional[List[int]] = None, language2: Optional[str] = None,
                  conflicting: bool = False) -> str:
    '''Get the file name for the file, incorporating layers, languages, and conflicts.'''

    training_dataset = "imdb_simplicity"
    layer_str = "all" if layers is None else f"{layers[0]}_{layers[-1]}"
    conflict_str = f"_conflict" if conflicting else ""
    language_str = f"{language}_{language2}" if language2 else language
    
    if file_type == 'simplicity_vector':
        return f'{training_dataset}_topic_vector_{language_str}_{num_tuples}_for_{config.MODEL_ALIAS}_{layer_str}{conflict_str}.pkl'
    if file_type == 'training_data':
        return f'{training_dataset}_train_{language}_{num_tuples}.json'
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

def train_simplicity_vectors(config: ExperimentConfig, language: str, num_tuples: int,
                            layers: Optional[List[int]], language2: Optional[str] = None,
                            conflicting: bool = False) -> Optional[Dict[str, float]]:
    """Train the simplicity vectors for a given language, potentially with conflicting simplicity."""
    tokenizer, model, _ = load_model_and_tokenizer(config=config, CustomModel=None)
    
    training_data = get_simplicity_training_data(config, language, num_tuples)
    training_tuples = [tuple(pair) for pair in training_data['tuples']]

    if language2:
        training_data2 = get_simplicity_training_data(config, language2, num_tuples)
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

def get_simplicity_training_data(config: ExperimentConfig, language: str, num_tuples: int
                                ) -> Optional[Dict]:
    """Get the training data, potentially with conflicting simplicity for two languages."""
    file_path = get_file_path(config, 'training_data', language, num_tuples)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.info("Training data not found. Generating it now.")
        return generate_and_store_training_data(config, language, num_tuples,
                                                dataset_name='imdb_simplicity_train')

def create_training_data(data: pd.DataFrame, num_tuples: int, max_length: Optional[int] = 200
                         ) -> List[Tuple[str, str]]:
    '''
    Create tuples of positive and negative samples from the dataset. Only use samples with 
    a token count of less than or equal to the max_length.

    :param data: The dataset to create the training data from
    :param num_tuples: The number of tuples to create
    :param max_length: The maximum token count for a sample
    :return: A list of tuples of positive and negative samples
    '''
    data['token_count'] = data['text'].apply(lambda x: len(x.split()))
    data_short = data[data['token_count'] <= max_length]
    
    positive_samples = data_short[data_short['label'] == 1].head(num_tuples)
    negative_samples = data_short[data_short['label'] == 0].head(num_tuples)
    
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

def generate_and_store_training_data(config: ExperimentConfig, languages: List[str], num_tuples: int,
                                     dataset_name: str) -> Optional[Dict]:
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
    """Train simplicity vectors for specified languages."""
    config = ExperimentConfig()
    NUM_TUPLES = 5
    LANGUAGES = ['en', 'de', 'fr']
    
    for language in LANGUAGES:
        simplicity_vector = get_simplicity_vector(config, language, NUM_TUPLES)
        if simplicity_vector:
            logger.info(f"Successfully processed simplicity vector for {language}")
        else:
            logger.error(f"Failed to process simplicity vector for {language}")

# pylint: enable=logging-fstring-interpolation
if __name__ == "__main__":
    main()