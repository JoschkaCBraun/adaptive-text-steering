"""
create_sentiment_vector_training_samples.py

This file contains the code to create training samples for the sentiment vector.

Text is loaded from datasets/sentiment/sentiment_synthetic_data/ 
Options are sentiment_sentences and sentiment_words.
Return a list of tuples of positive and negative samples.
"""
import os
import json
import logging
from typing import Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd
from transformers import pipeline
from datasets import Dataset

from src.utils import get_file_path, get_device, pipeline, logger, load_sentiment_representations


def create_sentiment_vector_training_samples(config: ExperimentConfig, language: str, num_tuples: int, representation_type: str) -> Optional[Dict]:
    '''
    Create training samples for the sentiment vector.
    '''
    # validate inputs
    validate_language(language)
    if num_tuples <= 0 or num_tuples > 500:
        raise ValueError(f"Invalid number of tuples: {num_tuples}")

    # load data
    data = load_sentiment_representations(representation_type)
    

def get_sentiment_training_data(config: ExperimentConfig, language: str) -> Optional[Dict]:
    """Get the training data, potentially with conflicting sentiment for two languages."""
    file_path = get_file_path(config, 'training_data', language, num_tuples)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.info("Training data not found. Generating it now.")
        return generate_and_store_training_data(config, language, num_tuples,
                                                dataset_name='imdb_sentiment_train')

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
    """Train sentiment vectors for specified languages."""
    config = ExperimentConfig()
    NUM_TUPLES = 5
    LANGUAGES = ['en', 'de', 'fr']
    
    for language in LANGUAGES:
        sentiment_vector = get_sentiment_vector(config, language, NUM_TUPLES)
        if sentiment_vector:
            logger.info(f"Successfully processed sentiment vector for {language}")
        else:
            logger.error(f"Failed to process sentiment vector for {language}")
