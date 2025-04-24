"""
create_sentiment_vector_training_samples.py

This file contains the code to create training samples for the sentiment vector.

Text is loaded from datasets/sentiment/sentiment_synthetic_data/ 
Options are sentiment_sentences and sentiment_words.
Return a list of tuples of positive and negative samples.
"""

from typing import List, Tuple

from src.utils import load_sentiment_vector_training_samples

def create_sentiment_vector_training_samples(num_samples: int, representation_type: str, language: str = 'en') -> List[Tuple[str, str]]:
    '''
    Create training samples for the sentiment vector.
    '''
    training_samples = load_sentiment_vector_training_samples(num_samples, representation_type, language)
    return training_samples
