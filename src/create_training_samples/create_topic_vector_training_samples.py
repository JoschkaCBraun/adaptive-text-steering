'''
create_training_samples.py

This script creates training samples for all relevant topic ids in the dataset.
'''
# Standard library imports
import logging
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd
import random
from src.utils import load_topic_representations, validate_topic_representation_type, validate_pairing_type, save_topic_vector_training_samples
from config.experiment_config import ExperimentConfig

'''
I want to load the topic representations from the json files and then create training samples
from them.

There should be three types or training samples: 
1. topic words
2. topic phrases
3. topic descriptions
4. topic summaries

And the pairing should have the options: 
topic representation vs random topic representation
topic representation vs random string

Every function generating training samples should take in a tid and a number of training samples to generate.
'''

# Set up logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_all_topic_vector_training_samples(config: ExperimentConfig) -> None:
    '''
    Create all topic vector training samples for all topics and topic representations and for the pairing type "against_random_topic_representation".
    '''
    for topic_representation_type in config.TOPIC_REPRESENTATION_TYPES:
        num_samples = config.TOPIC_REPRESENTATION_TYPE_NUM_SAMPLES[topic_representation_type]
        for tid in config.TOPIC_IDS:
            create_topic_vector_training_samples(config, topic_representation_type, "against_random_topic_representation", num_samples, tid)

def create_topic_vector_training_samples(config: ExperimentConfig, topic_representation_type: str, pairing_type: str, num_samples: int, tid: int
                                         ) -> List[Tuple[str, str]]:
    '''Create num_samples training samples from topic representation type to train topic vector 
    with pairing_type for tid.

    :param config: A Config object containing the configuration parameters.
    :param topic_representation_type: The type of topic representation to use.
    :param pairing_type: The type of pairing to use.
    :param num_samples: The number of training samples to generate.
    :param tid: The topic id to generate training samples for.
    :return: A list of tuples of (positive_prompt, negative_prompt).
    '''
    # validate inputs
    validate_topic_representation_type(topic_representation_type)
    validate_pairing_type(pairing_type)

    topic_representations = load_topic_representations(topic_representation_type)

    training_samples = create_training_pairs(topic_representations[tid], topic_representations,
                                             tid, num_samples)

    save_topic_vector_training_samples(training_samples, topic_representation_type, pairing_type, tid)
    return training_samples

def create_training_pairs(positive_samples: List[str], all_samples: Dict[int, List[str]], 
                          tid_positive: int, num_samples: int) -> List[Tuple[str, str]]:
    """
    Creates training pairs by combining positive samples with negative samples.
    Negative samples are collected in an 'interleaved' order from all
    negative topics.

    If num_samples is less than the total number of possible unique pairs,
    a random subsample of that size is returned. Otherwise, all unique
    pairs are returned.

    :param positive_samples: List of samples for the positive topic.
    :param all_samples: Dictionary of all samples, Key=Topic ID, Value=List of samples.
    :param tid_positive: The ID of the positive topic.
    :param num_samples: The desired number of training pairs.

    :return: A list of (positive_sample, negative_sample) tuples.

    :raises: ValueError: If no positive samples or no non-empty negative sample lists
                    are found for the given tid_positive.
    """

    # --- 1. Collect negative samples in an interleaved order ---
    negative_lists = [
        negs for tid, negs in all_samples.items()
        if tid != tid_positive and negs # Only consider topics with actual samples
    ]

    # Validate inputs
    if not positive_samples:
        logger.error(f"No positive samples found for topic ID {tid_positive}.")
        raise ValueError(f"No positive samples found for topic ID {tid_positive}.")
    if not negative_lists:
         logger.error(f"No non-empty negative sample lists found for topic ID {tid_positive}.")
         raise ValueError(f"No non-empty negative sample lists found for topic ID {tid_positive}.")
    if num_samples < 0:
        logger.warning(f"num_samples ({num_samples}) is negative. Returning empty list.")
        return []

    interleaved_negatives = []
    # Find the maximum length among negative lists
    max_len =  max(len(neg_list) for neg_list in negative_lists)

    # Build the interleaved list
    for i in range(max_len):
        for neg_list in negative_lists:
            if i < len(neg_list):  # Check if the list has an element at this index
                interleaved_negatives.append(neg_list[i])

    # --- 2. Generate all possible unique combinations ---
    all_pairs = []
    for pos_sample in positive_samples:
        for neg_sample in interleaved_negatives:
            all_pairs.append((pos_sample, neg_sample))

    total_possible_pairs = len(all_pairs)
    # --- 3. Handle num_samples: Subsample or return all ---
    if num_samples <= total_possible_pairs:
        return random.sample(all_pairs, num_samples)
    else:
        logger.warning(f"Requested num_samples ({num_samples}) is greater than total possible pairs ({total_possible_pairs}). "
                       f"Returning all {total_possible_pairs} unique pairs.")
        return all_pairs

def create_training_samples(config: ExperimentConfig, topic_representation_type: str
                          ) -> Dict[int, List[Tuple[str, str]]]:
    """Create training samples using stored data."""
    training_samples: Dict[int, List[Tuple[str, str]]] = {}
    
    if topic_representation_type == 'topic_strings':
        # Load all types
        words = load_topic_representations("topic_words")
        phrases = load_topic_representations("topic_phrases")
        descriptions = load_topic_representations("topic_descriptions")
        
        # Create pairs for each topic
        for tid in words.keys():
            training_samples[tid] = []
            # Add pairs from each type
            training_samples[tid].extend(create_training_pairs(words[tid], words, tid, config.NUM_SAMPLES))
            training_samples[tid].extend(create_training_pairs(phrases[tid], phrases, tid, config.NUM_SAMPLES))
            training_samples[tid].extend(create_training_pairs(descriptions[tid], descriptions, tid, config.NUM_SAMPLES))
            
    elif topic_representation_type == 'topic_summaries':
        summaries = load_topic_representations("topic_summaries")
        for tid in summaries.keys():
            training_samples[tid] = create_training_pairs(summaries[tid], summaries, tid, config.NUM_SAMPLES)
            
    return training_samples

def create_samples_from_strings(tid_positive: int, topic_strings: Dict[int, Dict[str, List[str]]],
                                ) -> List[Tuple[str, str]]:
    '''Create training samples from topic strings to train topic vector for tid_pos.

    :param tid_positive: The positive topic id for which the steering vector is to be trained.
    :param topic_strings: A dictionary of topic strings for all relevant topic ids.
    :return: A list of tuples of (positive_prompt, negative_prompt).
    '''
    training_samples = []

    for tid_negative, negative_data in topic_strings.items():
        if tid_negative == tid_positive:
            continue
        for key, positive_data in topic_strings[tid_positive].items():
            training_samples.append((' '.join(positive_data), ' '.join(negative_data[key])))

    return training_samples

def create_samples_from_summaries(config: ExperimentConfig, tid_positive: int, df: pd.DataFrame
                                  ) -> List[Tuple[str, str]]:
    ''' Create training samples from summaries to train the steering vector for tid_positive.
    
    :param config: A Config object containing the configuration parameters.
    :param tid_positive: The positive topic id for which the steering vector is to be trained.
    :param df: A pandas DataFrame containing the dataset.
    :return: A list of tuples of (positive_prompt, negative_prompt).    
    '''
    training_samples = []

    include_non_matching = config.INCLUDE_NON_MATCHING
    num_samples = config.NUM_SAMPLES
    
    # Filter summaries specifically associated with the positive topic id
    positive_summaries = df.loc[df['tid1'] == tid_positive, 'summary1'].tolist() + \
                         df.loc[df['tid2'] == tid_positive, 'summary2'].tolist()

    len_positive_summaries = len(positive_summaries)
    if len_positive_summaries == 0:
        logger.error(f"No summaries found for topic id {tid_positive}")
        raise ValueError(f"No summaries found for topic id {tid_positive}")
    
    non_matching_samples_needed = max(0, num_samples - len_positive_summaries)
    pos_idx = 0
    
    for _, row in df.iterrows():
        if row['tid1'] == tid_positive:
            training_samples.append((row['summary1'], row['summary2']))

        elif row['tid2'] == tid_positive:
            training_samples.append((row['summary2'], row['summary1']))

        elif include_non_matching and non_matching_samples_needed > 0:
            training_samples.append((positive_summaries[pos_idx % len_positive_summaries],
                                     row['summary1']))
            non_matching_samples_needed -= 1
            if non_matching_samples_needed > 0:
                training_samples.append((positive_summaries[(pos_idx + 1) % len_positive_summaries],
                                         row['summary2']))
                non_matching_samples_needed -= 1
            pos_idx += 2

    return training_samples


def main() -> None:
    config = ExperimentConfig()
    create_all_topic_vector_training_samples(config)

if __name__ == "__main__":
    main()

