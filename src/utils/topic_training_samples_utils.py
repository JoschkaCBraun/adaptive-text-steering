"""
Utilities for managing topic training samples.
"""
import os
import json
from datetime import datetime
from typing import Dict, List

def save_topic_training_samples(samples: Dict[int, List[str]], 
                         sample_type: str,
                         language: str = 'en') -> None:
    """
    Save training samples to a JSON file.
    
    :param samples: Dictionary mapping topic IDs to lists of positive samples
    :param sample_type: Type of samples (topic_words, topic_phrases, topic_descriptions, topic_summaries)
    :param language: Language code for the samples
    """
    # Create directory if it doesn't exist
    samples_dir = os.getenv('TOPIC_TRAINING_SAMPLES_PATH')
    os.makedirs(samples_dir, exist_ok=True)
    
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
            "type": sample_type,
            "language": language
        },
        "samples": {str(tid): samples[tid] for tid in samples}
    }
    
    # Save to file
    filename = make_topic_training_samples_file_name(sample_type, language)
    filepath = os.path.join(samples_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_topic_training_samples(sample_type: str, 
                         language: str = 'en') -> Dict[int, List[str]]:
    """
    Load training samples from a JSON file.
    
    :param sample_type: Type of samples to load
    :param language: Language code for the samples
    :return: Dictionary mapping topic IDs to lists of positive samples
    """
    samples_dir = os.getenv('TOPIC_TRAINING_SAMPLES_PATH')
    filename = make_topic_training_samples_file_name(sample_type, language)
    filepath = os.path.join(samples_dir, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert string keys back to integers
            return {int(tid): samples for tid, samples in data["samples"].items()}
    except FileNotFoundError:
        raise FileNotFoundError(f"Training samples file not found: {filepath}")
    

def make_topic_training_samples_file_name(sample_type: str, language: str) -> str:
    """
    Make a file name from sample type and language.
    """
    return f"topic_{sample_type}_{language}_training_samples.json"
