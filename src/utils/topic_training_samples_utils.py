"""
Utilities for managing topic training samples.
"""
import os
import json
from datetime import datetime
from typing import Dict, List

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
    
def get_topic_representations_file_name(representation_type: str, language: str) -> str:
    """
    Make a file name from representation type and language.
    """
    return f"{representation_type}_{language}_representations.json"
