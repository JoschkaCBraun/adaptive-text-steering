'''
create_training_samples.py

This script creates training samples for all relevant topic ids in the dataset.
'''
# Standard library imports
import logging
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd

from src.utils import load_topic_representations, validate_topic_representation_type, validate_pairing_type
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

def create_topic_vector_training_samples(config: ExperimentConfig, topic_representation_type: str,
                                         pairing_type: str, num_samples: int, tid: int
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

    # load topic representations
    topic_representations = load_topic_representations(topic_representation_type)

    # create training samples
    training_samples = create_training_pairs(topic_representations[tid], topic_representations,
                                             tid, num_samples)
    return training_samples

def create_training_pairs(positive_samples: List[str], all_samples: Dict[int, List[str]], 
                          tid_positive: int, num_samples: int) -> List[Tuple[str, str]]:
    """Create training pairs from samples using a round-robin approach."""
    pairs = []
    negative_samples = [neg for tid, negs in all_samples.items() if tid != tid_positive for neg in negs]
    
    pos_len = len(positive_samples)
    neg_len = len(negative_samples)
    
    for i in range(num_samples):
        pos_index = i % pos_len
        neg_index = i % neg_len
        pairs.append((positive_samples[pos_index], negative_samples[neg_index]))
    
    return pairs

def store_training_samples(config: ExperimentConfig, topic_encoding_type: str,
                          training_samples: Dict[int, List[Tuple[str, str]]]) -> None:
    '''Store training samples in a json file.
    '''
    pass

def create_training_samples(config: ExperimentConfig, topic_encoding_type: str
                          ) -> Dict[int, List[Tuple[str, str]]]:
    """Create training samples using stored data."""
    training_samples = {}
    
    if topic_encoding_type == 'topic_strings':
        # Load all types
        words = load_training_samples("topic_words")
        phrases = load_training_samples("topic_phrases")
        descriptions = load_training_samples("topic_descriptions")
        
        # Create pairs for each topic
        for tid in words.keys():
            training_samples[tid] = []
            # Add pairs from each type
            training_samples[tid].extend(create_training_pairs(words[tid], words, tid))
            training_samples[tid].extend(create_training_pairs(phrases[tid], phrases, tid))
            training_samples[tid].extend(create_training_pairs(descriptions[tid], descriptions, tid))
            
    elif topic_encoding_type == 'topical_summaries':
        summaries = load_training_samples("topic_summaries")
        for tid in summaries.keys():
            training_samples[tid] = create_training_pairs(summaries[tid], summaries, tid)
            
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


# make main function that creates training samples for all topics and topic representations
def main():
    config = ExperimentConfig()
    create_training_samples(config)


if __name__ == "__main__":
    main()

