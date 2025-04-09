'''
create_topic_representations.py

This script creates topic representations for all relevant topic ids in the dataset.
'''
# Standard library imports
import logging
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd

from src.utils import load_lda, get_topic_words, load_newts_dataframe, save_topic_representations
from config.experiment_config import ExperimentConfig

# Set up logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_topic_representations(config: ExperimentConfig, topic_representation_type: str
                            ) -> Dict[int, List[Tuple[str, str]]]:
    '''Create training samples for the steering vector.

    :param topic_representation_type: The type of topic representation to use.
    :return: A dictionary of topic representations for each topic.
    '''
    df = load_newts_dataframe(num_articles=config.NUM_ARTICLES)
    topic_representations = dict()

    if topic_representation_type == 'topic_strings':
        topic_strings = generate_topic_strings(config=config, df=df)
        for tid in topic_strings.keys():
            topic_representations[tid] = create_samples_from_strings(tid_positive=tid,
                                                                topic_strings=topic_strings)
    elif topic_representation_type == 'topic_summaries':
        relevant_tids = list(map(int, df['tid1'].unique()))
        for tid in relevant_tids:
            topic_representations[tid] = create_samples_from_summaries(config=config, tid_positive=tid,
                                                                  df=df)
    else:
        logger.error(f"Invalid topic representation type: {topic_representation_type}")
        raise ValueError(f"Invalid topic representation type: {topic_representation_type}")
    
    return topic_representations

def generate_topic_strings(config: ExperimentConfig, df: pd.DataFrame) -> Dict[int, Dict[str, List[str]]]:
    '''Generate topic strings for all relevant topic ids in the dataset.

    :param config: A Config object containing the configuration parameters.
    :param df: A pandas DataFrame containing the dataset.
    :return: A dictionary of topic strings for all relevant topic ids.
    '''
    relevant_tids = set(map(int, df['tid1'].unique())).union(set(map(int, df['tid2'].unique())))
    topic_strings: Dict[int, Dict[str, List[str]]] = {tid: {} for tid in relevant_tids}
    lda = load_lda()
    num_topic_words = config.NUM_TOPIC_WORDS
    
    for tid in relevant_tids:
        # Store topic words as individual strings in a list
        topic_strings[tid]['topic_words'] = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
    
    missing_tids = set(relevant_tids)
    for _, row in df.iterrows():
        tid1, tid2 = int(row['tid1']), int(row['tid2'])
        if tid1 in missing_tids:
            # Split phrases by comma and strip whitespace
            topic_strings[tid1]['topic_phrases'] = [phrase.strip() for phrase in row['phrases1'].split(',')]
            # Store the entire description as a single string in a list
            topic_strings[tid1]['topic_description'] = [row['sentences1']]
            missing_tids.remove(tid1)
        if tid2 in missing_tids:
            topic_strings[tid2]['topic_phrases'] = [phrase.strip() for phrase in row['phrases2'].split(',')]
            topic_strings[tid2]['topic_description'] = [row['sentences2']]
            missing_tids.remove(tid2)
        if len(missing_tids) == 0:
            break

    return topic_strings

def generate_and_save_all_samples(config: ExperimentConfig) -> None:
    """Generate and save all types of training samples.
    Note: Although we use integer keys in our dictionaries, they will be converted
    to strings when saved as JSON, since JSON only supports string keys.
    """
    df = load_newts_dataframe(num_articles=config.NUM_ARTICLES)
    
    # Get all topic strings
    topic_strings = generate_topic_strings(config=config, df=df)
    
    # Prepare separate collections for each type
    topic_words: Dict[int, List[str]] = {}
    topic_phrases: Dict[int, List[str]] = {}
    topic_descriptions: Dict[int, List[str]] = {}
    topic_summaries: Dict[int, List[str]] = {}
    
    # Extract topic words, phrases, and descriptions
    for tid, data in topic_strings.items():
        tid = int(tid)  # Ensure tid is integer
        topic_words[tid] = data['topic_words']  # Already a list of individual words
        topic_phrases[tid] = data['topic_phrases']  # Already a list of phrases
        topic_descriptions[tid] = data['topic_description']  # Already a list with single description
    
    # Extract summaries
    for _, row in df.iterrows():
        for tid, summary in [(int(row['tid1']), row['summary1']), 
                           (int(row['tid2']), row['summary2'])]:
            if tid not in topic_summaries:
                topic_summaries[tid] = []
            topic_summaries[tid].append(summary)
    
    # Save all types
    save_topic_representations(topic_words, "topic_words")
    save_topic_representations(topic_phrases, "topic_phrases")
    save_topic_representations(topic_descriptions, "topic_descriptions")
    save_topic_representations(topic_summaries, "topic_summaries")

if __name__ == "__main__":
    config = ExperimentConfig()
    generate_and_save_all_samples(config)