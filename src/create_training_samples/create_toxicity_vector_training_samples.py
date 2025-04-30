"""
create_toxicity_vector_training_samples.py

This file contains the code to create training samples for the toxicity vector.

Text is loaded from datasets/toxicity/toxicity_synthetic_data/
Options are toxicity_sentences and toxicity_words.
Return a list of tuples of positive and negative samples.
"""

from typing import List, Tuple
import json
import random
import logging
import csv
logger = logging.getLogger(__name__)


def create_toxicity_vector_training_samples(num_samples: int, representation_type: str, language: str = 'en') -> List[Tuple[str, str]]:
    '''
    Create training samples for the toxicity vector.
    '''
    # 1. load the data depending on the representation type
    if representation_type == 'sentences':
        training_samples = load_toxicity_sentences(num_samples, language)
    elif representation_type == 'words':
        training_samples = load_toxicity_words(num_samples, language)
    else:
        raise ValueError(f"Invalid representation type: {representation_type}")
    return training_samples

def load_toxicity_sentences(num_samples: int, language: str) -> List[Tuple[str, str]]:
    # toxicity_sentences are stored in toxicity_sentences_synthetic_500_en.json
    if language != 'en':
        raise ValueError(f"Data not available for language: {language}")
    if num_samples > 500:
        raise ValueError(f"Data only available for 500 samples, got {num_samples}")
    
    toxicity_tuples : List[Tuple[str, str]] = []
    path = 'datasets/toxicity/toxicity_synthetic_data/toxicity_sentences_synthetic_500_en.csv'
    
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            toxic = row['toxic']
            non_toxic = row['non-toxic']
            toxicity_tuples.append((toxic, non_toxic))
    
    return toxicity_tuples

def load_toxicity_words(num_samples: int, language: str) -> List[Tuple[str, str]]:
    # toxicity_words are stored in toxicity_words_synthetic_100_en.json
    if language != 'en':
        raise ValueError(f"Data not available for language: {language}")
    toxicity_tuples: List[Tuple[str, str]] = []
    path = 'datasets/toxicity/toxicity_synthetic_data/toxicity_words_synthetic_100_en.json'
    with open(path, 'r') as f:
        toxicity_words = json.load(f)

    # dictionary with keys "toxic", "non_toxic", "neutral"
    # will output a 
    toxic_words = toxicity_words['toxic']
    non_toxic_words = toxicity_words['non_toxic']
    neutral_words = toxicity_words['neutral']

    if num_samples > 100:
        logger.info(f"Data only available for 100 samples, got {num_samples}, mixing all data.")
        # iterate over toxic words (in a loop) and select a random non-toxic word.
        # repeat until num_samples is reached
        toxicity_tuples = []
        idx = 0
        while idx < num_samples:
            toxic_word = toxic_words[idx % len(toxic_words)]
            non_toxic_word = random.choice(non_toxic_words)
            toxicity_tuples.append((toxic_word, non_toxic_word))
            idx += 1
    else:
        # sample num_samples from each category
        toxic_samples = toxic_words[:num_samples]
        non_toxic_samples = non_toxic_words[:num_samples]
        neutral_samples = neutral_words[:num_samples]
        # return tuples of toxic, non_toxic samples 
        toxicity_tuples = [(toxic, non_toxic) for toxic, non_toxic in zip(toxic_samples, non_toxic_samples)]

    return toxicity_tuples

def main() -> None:
    """
    Main function to test the create_toxicity_vector_training_samples functionality.
    """
    # Test with sentences representation
    print("Testing with sentences representation:")
    try:
        sentence_samples = create_toxicity_vector_training_samples(5, 'sentences')
        print(f"Generated {len(sentence_samples)} sentence samples")
        for i, (toxic, non_toxic) in enumerate(sentence_samples):
            print(f"Sample {i+1}:")
            print(f"  Toxic: {toxic}")
            print(f"  Non-toxic: {non_toxic}")
    except Exception as e:
        print(f"Error with sentences: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test with words representation
    print("Testing with words representation:")
    try:
        word_samples = create_toxicity_vector_training_samples(5, 'words')
        print(f"Generated {len(word_samples)} word samples")
        for i, (toxic, non_toxic) in enumerate(word_samples):
            print(f"Sample {i+1}:")
            print(f"  Toxic: {toxic}")
            print(f"  Non-toxic: {non_toxic}")
    except Exception as e:
        print(f"Error with words: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test error handling - invalid representation type
    print("Testing error handling - invalid representation type:")
    try:
        invalid_samples = create_toxicity_vector_training_samples(5, 'invalid_type')
        print("This should not be reached")
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    # Test error handling - too many samples
    print("\nTesting error handling - too many samples:")
    try:
        too_many_samples = create_toxicity_vector_training_samples(600, 'sentences')
        print("This should not be reached")
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    # Test error handling - unsupported language
    print("\nTesting error handling - unsupported language:")
    try:
        unsupported_language = create_toxicity_vector_training_samples(5, 'sentences', language='fr')
        print("This should not be reached")
    except ValueError as e:
        print(f"Expected error caught: {e}")


if __name__ == "__main__":
    main()



