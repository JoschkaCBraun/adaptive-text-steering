'''
get_steering_vector_file_paths.py

This module provides utilities for generating standardized file paths for
pre-computed toxicity and readability steering vectors. It helps ensure
consistent naming and organization based on vector parameters.
'''

import os
from typing import List, Literal
import pickle

from steering_vectors import SteeringVector

from config.experiment_config import ExperimentConfig

import logging
logger = logging.getLogger(__name__)


# --- Constants ---
# behaviours 

config = ExperimentConfig()

VALID_BEHAVIOR_TYPES = config.VALID_BEHAVIOR_TYPES

# topic vector
TOPIC_WORDS = 'words'
TOPIC_PHRASES = 'phrases'
TOPIC_DESCRIPTIONS = 'descriptions'
TOPIC_SUMMARIES = 'summaries'
VALID_TOPIC_REPRESENTATION_TYPES = {TOPIC_WORDS, TOPIC_PHRASES, TOPIC_DESCRIPTIONS, TOPIC_SUMMARIES}
# toxicity vector
TOXICITY_SENTENCES = 'sentences'
TOXICITY_WORDS = 'words'
VALID_TOXICITY_REPRESENTATION_TYPES = {TOXICITY_SENTENCES, TOXICITY_WORDS}

# readability vector
READABILITY_SENTENCES = 'sentences'
READABILITY_WORDS = 'words'
VALID_READABILITY_REPRESENTATION_TYPES = {READABILITY_SENTENCES, READABILITY_WORDS}

def load_steering_vector(
    behavior_type: str,
    model_alias: str,
    representation_type: Literal['sentences', 'words'],
    num_samples: int,
    steering_layers: List[int],
    language: str = 'en',
    pairing_type: str = None,
    tid: int = None,
) -> SteeringVector:
    '''
    Load the topic vector from a file.
    '''
    # check if file exists and if it is a steering vector
    topic_vector_file_path = get_steering_vector_path(
        behavior_type=behavior_type,
        model_alias=model_alias,
        representation_type=representation_type,
        num_samples=num_samples,
        steering_layers=steering_layers,
        language=language,
        pairing_type=pairing_type,
        tid=tid)
    if not os.path.exists(topic_vector_file_path):
        logger.info(f"File not found: {topic_vector_file_path}")
        return None
    with open(topic_vector_file_path, 'rb') as f:
        steering_vector = pickle.load(f)
    if not isinstance(steering_vector, SteeringVector):
        logger.info(f"File is not a steering vector: {topic_vector_file_path}")
        return None
    return steering_vector

def save_steering_vector(
    steering_vector: SteeringVector,
    behavior_type: str,
    model_alias: str,
    representation_type: Literal['sentences', 'words'],
    num_samples: int,
    steering_layers: List[int],
    language: str = 'en',
    pairing_type: str = None,
    tid: int = None,) -> None:
    '''
    Save the topic vector to a file.
    '''
    topic_vector_file_path = get_steering_vector_path(
        behavior_type=behavior_type,
        model_alias=model_alias,
        representation_type=representation_type,
        num_samples=num_samples,
        steering_layers=steering_layers,
        language=language,
        pairing_type=pairing_type,
        tid=tid)
    with open(topic_vector_file_path, 'wb') as f:
        pickle.dump(steering_vector, f)

def get_steering_vector_path(
    behavior_type: str,
    model_alias: str,
    representation_type: Literal['sentences', 'words'],
    num_samples: int,
    steering_layers: List[int],
    language: str = 'en',
    pairing_type: str = None,
    tid: int = None,
) -> str:
    """
    Constructs the standardized path for a toxicity steering vector file.

    Vectors are expected to be stored under:
    <base_path>/<TOXICITY_SUBDIR>/<generated_filename>.<vector_format>
    where <base_path> is determined by the 'STEERING_VECTOR_PATH' environment
    variable or the `base_path_override` argument.

    Args:
        behavior_type (str): Type of behavior the vector targets.
        model_alias (str): Name of the base model (e.g., 'llama3-8b', 'gpt-j-6b').
                          Used in the filename.
        representation_type (str): Type of data used for training the vector
                                   ('sentences' or 'words'). Used in the filename.
        num_samples (int): Number of training sample pairs used to create the vector.
                           Used in the filename.
        steering_layers (List[int]): List of layer numbers the vector targets.
            Formatted by `_format_layer_spec` for the filename.
        language (str): Language code for the vector (default: 'en'). Used in the filename.

    Returns:
        str: The full, standardized path intended for the steering vector file.

    Raises:
        ValueError: If the base path cannot be determined (env var not set and no override).
        ValueError: If representation_type is invalid.
        ValueError: If num_samples is not a positive integer.
        ValueError: If model_name is empty.
        ValueError: If layer_spec format is invalid (e.g., bad range, negative numbers, empty list).
        TypeError: If layer_spec has an unsupported type.
    """

    # 2. Validate Core Inputs
    if behavior_type not in VALID_BEHAVIOR_TYPES:
        raise ValueError(f"Invalid behavior type: '{behavior_type}'. "
                         f"Expected one of {VALID_BEHAVIOR_TYPES}")
    if behavior_type == "toxicity":
        if representation_type not in VALID_TOXICITY_REPRESENTATION_TYPES:
            raise ValueError(f"Invalid representation type: '{representation_type}'. "
                            f"Expected one of {VALID_TOXICITY_REPRESENTATION_TYPES}")
    elif behavior_type == "readability":
        if representation_type not in VALID_READABILITY_REPRESENTATION_TYPES:
            raise ValueError(f"Invalid representation type: '{representation_type}'. "
                            f"Expected one of {VALID_READABILITY_REPRESENTATION_TYPES}")
    elif behavior_type == "topic":
        if representation_type not in VALID_TOPIC_REPRESENTATION_TYPES:
            raise ValueError(f"Invalid representation type: '{representation_type}'. "
                            f"Expected one of {VALID_TOPIC_REPRESENTATION_TYPES}")
        if pairing_type is None:
            raise ValueError("pairing_type cannot be None for topic vectors.")
        if tid is None:
            raise ValueError("tid cannot be None for topic vectors.")
    if not isinstance(num_samples, int) or num_samples <= 0:
         raise ValueError(f"num_samples must be a positive integer, got {num_samples}")
    if not model_alias:
        raise ValueError("model_alias cannot be empty.")
    if not language:
        raise ValueError("language cannot be empty.")

    # 4. Construct Filename
    # Example: llama3-8b_toxicity_sentences_500samples_layer_15_en_v1.pt
    if behavior_type in {"toxicity", "readability"}:
        filename = f"{behavior_type}_{model_alias}_layer_{steering_layers[0]}_{representation_type}_{language}_{num_samples}_samples.pkl"
    elif behavior_type == "topic":
        filename = f"{behavior_type}_{model_alias}_layer_{steering_layers[0]}_{representation_type}_{pairing_type}_tid_{tid}_{language}_{num_samples}_samples.pkl"
    else:
        raise ValueError(f"Invalid behavior type: {behavior_type}")

    # 5. Combine Path Components
    folder_path = get_steering_vectors_folder_path(behavior_type=behavior_type,
                                                   representation_type=representation_type)
    full_path = os.path.join(folder_path, filename)
    return full_path

def get_steering_vectors_folder_path(behavior_type: str, representation_type: str) -> str: 
    '''Get the folder path for storing topic vectors.'''
    steering_vectors_path = os.getenv('STEERING_VECTOR_PATH')
    if steering_vectors_path is None:
        raise ValueError("STEERING_VECTOR_PATH is not set")
    folder_path = os.path.join(steering_vectors_path, f"{behavior_type}_vectors", representation_type)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# --- Main Function for Demonstration ---
def main():
    """Demonstrates the usage of get_steering_vector_path."""
    print("\n" + "=" * 60)
    print(" Demonstrating Toxicity Steering Vector Path Generation")

    base_path = os.getenv('STEERING_VECTOR_PATH')

    try:
        # --- Basic Examples ---
        print("\n--- Basic Examples ---")
        path1 = get_steering_vector_path(
            behavior_type="toxicity",
            model_alias='llama3-8b',
            representation_type=TOXICITY_SENTENCES,
            num_samples=500,
            steering_layers=[15], # Single layer int
            language='en'
        )

        path2 = get_steering_vector_path(
            behavior_type="toxicity",
            model_alias='mistral-7b',
            representation_type=TOXICITY_WORDS,
            num_samples=40,
            steering_layers=[10, 11, 12, 13, 14], # Layer range string
            language='en',
        )

        # --- Language and Format ---
        print("\n--- Language and Format Examples ---")
        path3 = get_steering_vector_path(
            behavior_type="toxicity",
            model_alias='gemma-7b',
            representation_type=TOXICITY_SENTENCES,
            num_samples=500,
            steering_layers=[12, 13, 14], # Contiguous layer list
            language='de', # Different language
        )

        path4 = get_steering_vector_path(
            behavior_type="toxicity",
            model_alias='llama3-70b',
            representation_type=TOXICITY_SENTENCES,
            num_samples=500,
            steering_layers=[12, 13, 14], # 'all' layers keyword
            language='multi' # Hypothetical multilingual vector
        )

        # --- Advanced Layer Specs ---
        print("\n--- Advanced Layer Specification Examples ---")
        path5 = get_steering_vector_path(
            behavior_type="toxicity",
            model_alias='llama3-8b',
            representation_type=TOXICITY_SENTENCES,
            num_samples=500,
            steering_layers=[8, 12, 16], # Non-contiguous layer list
        )

        # --- Readability Vector Examples ---
        print("\n--- Readability Vector Examples ---")
        path6 = get_steering_vector_path(
            behavior_type="readability",
            model_alias='llama3-8b',
            representation_type=READABILITY_SENTENCES,
            num_samples=500,
            steering_layers=[15],
            language='en'
        )
        
        path7 = get_steering_vector_path(
            behavior_type="readability",
            model_alias='mistral-7b',
            representation_type=READABILITY_WORDS,
            num_samples=40,
            steering_layers=[10, 11, 12, 13, 14],
            language='en'
        )

    except ValueError as e:
        print(f"\nERROR during demonstration: {e}")
    except TypeError as e:
        print(f"\nTYPE ERROR during demonstration: {e}")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
    finally:
        print(" Demonstration finished.")
        print("=" * 60)


if __name__ == "__main__":
    main()
