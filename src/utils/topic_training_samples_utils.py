"""
Utilities for managing topic representations and training samples.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

def save_topic_representations(samples: Dict[int, List[str]], representation_type: str, language: str = 'en') -> None:
    """
    Save topic representations to a JSON file.
    
    :param samples: Dictionary mapping topic IDs to lists of positive samples
    :param representation_type: Type of representations (topic_words, topic_phrases, topic_descriptions, topic_summaries)
    :param language: Language code for the samples
    """
    # Create directory if it doesn't exist
    representations_dir = os.getenv('TOPIC_REPRESENTATIONS_PATH')
    os.makedirs(representations_dir, exist_ok=True)
    
    # Validate input data
    for tid, sample_list in samples.items():
        if not isinstance(tid, int):
            raise TypeError(f"Topic ID must be an integer, got {type(tid)}")
        if not isinstance(sample_list, list):
            raise TypeError(f"Samples for topic {tid} must be a list, got {type(sample_list)}")
        for sample in sample_list:
            if not isinstance(sample, str):
                raise TypeError(f"Sample in topic {tid} must be a string, got {type(sample)}")
    
    # Prepare and fill data structure
    data = {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "type": representation_type,
            "language": language
        },
        "samples": {str(tid): samples[tid] for tid in samples}
    }
    
    # Save to file
    filename = get_topic_representations_file_name(representation_type, language)
    filepath = os.path.join(representations_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_topic_representations(representation_type: str, language: str = 'en'
                               ) -> Dict[int, List[str]]:
    """
    Load representations from a JSON file.
    
    :param representation_type: Type of representations to load
    :param language: Language code for the representations
    :return: Dictionary mapping topic IDs to lists of representations
    """
    representations_dir = os.getenv('TOPIC_REPRESENTATIONS_PATH')
    filename = get_topic_representations_file_name(representation_type, language)
    filepath = os.path.join(representations_dir, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert string keys back to integers
            return {int(tid): samples for tid, samples in data["samples"].items()}
    except FileNotFoundError:
        raise FileNotFoundError(f"Representations file not found: {filepath}")
    
def get_topic_representations_file_name(representation_type: str, language: str = 'en') -> str:
    """
    Make a file name from representation type and language.
    """
    return f"topic_{representation_type}_{language}_representations.json"

def save_topic_vector_training_samples(samples: Dict[int, List[Tuple[str, str]]], representation_type: str, pairing_type: str, tid: int, language: str = 'en') -> None:
    '''
    Save the topic vector training samples to a JSON file.
    '''
    file_path = get_topic_vector_training_samples_file_path(representation_type, pairing_type, tid, language)
    # save the samples to the file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
def load_topic_vector_training_samples(representation_type: str, pairing_type: str, tid: int, language: str = 'en') -> List[Tuple[str, str]]:
    '''
    Load the topic vector training samples from a JSON file.
    
    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains (positive_sample, negative_sample)
    '''
    file_path = get_topic_vector_training_samples_file_path(representation_type, pairing_type, tid, language)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [tuple(pair) for pair in data]

def get_topic_vector_training_samples_file_path(representation_type: str, pairing_type: str, tid: int, language: str = 'en') -> str:
    '''
    Get the file path for the topic vector training samples.
    '''
    file_name = get_topic_vector_training_samples_file_name(representation_type, pairing_type, tid, language)
    path = os.getenv('TOPIC_VECTOR_TRAINING_SAMPLES_PATH')
    if path is None:
        raise ValueError("TOPIC_VECTOR_TRAINING_SAMPLES_PATH is not set")
    return os.path.join(path, f"topic_{representation_type}", file_name)

def get_topic_vector_training_samples_file_name(representation_type: str, pairing_type: str, tid: int, language: str = 'en') -> str:
    '''
    Get the file name for the topic vector training samples.
    '''
    return f"tid_{tid}_topic_{representation_type}_{pairing_type}_{language}_training_samples.json"