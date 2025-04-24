"""
lda_utils.py   

Utility functions for loading the LDA model and dictionary.
"""
# Standard library imports
import os
import logging
from typing import Optional, Tuple, List

# Third-party imports
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from transformers import AutoTokenizer

# Local imports
from config.experiment_config import ExperimentConfig
from src.utils.text_processing_utils import TextProcessor
from src.utils.get_basic import get_path

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation

def load_lda_and_dictionary(return_lda: bool=True, return_dictionary: bool=True) -> Tuple:
    """
    Loads and returns the LDA model and dictionary from the filesystem.
    
    Returns:
    - tuple: (LdaModel or None, Dictionary or None) depending on success or failure of loading.
    """
    data_path = get_path("DATA_PATH")
    if not data_path:
        logging.error("Data directory not found. Did not load the LDA model or dictionary.")
        return None, None

    lda_model_path = os.path.join(data_path, "LDA", 'lda.model')
    dictionary_path = os.path.join(data_path, "LDA", 'dictionary.dic')
    lda, dictionary = None, None

    try:
        if return_lda:
            lda = LdaModel.load(lda_model_path, mmap='r')
            logging.info("Successfully loaded the LDA model.")
        if return_dictionary:
            dictionary = Dictionary.load(dictionary_path, mmap='r')
            logging.info("Successfully loaded the dictionary.")
        return lda, dictionary

    except Exception as e:
        logging.error(f"Error loading the LDA model or dictionary: {e}")
        return None, None

def load_dictionary() -> Dictionary:
    """Loads and returns the dictionary from the filesystem."""
    _, dictionary = load_lda_and_dictionary(return_lda=False, return_dictionary=True)
    return dictionary

def load_lda() -> LdaModel:
    """
    Loads and returns the LDA model from the filesystem.
    """
    lda, _ = load_lda_and_dictionary(return_lda=True, return_dictionary=False)
    return lda

def get_topic_words(lda: LdaModel, tid: int, num_topic_words: Optional[int] = None,
                    config: Optional[ExperimentConfig] = None) -> List[str]:
    """
    Returns the top words for a given topic from the LDA model.
    
    :param lda: The trained LDA model.
    :param tid: The topic number to get the top words for.
    :param num_topic_words: The number of top words to return for the specified topic, if any,
                            otherwise the default number of topic words is used.
    :param config: The configuration object. Required if num_topic_words is not provided.
    :return: A list of top words for the specified topic.
    """
    try:
        # Note: num_topic_words here limits the number of words returned for the topic.
        if num_topic_words is None and config is None:
            logging.error("Either num_topic_words or config is required to get top topic words.")
            return []
        num_topic_words = config.NUM_TOPIC_WORDS if num_topic_words is None else num_topic_words
        topic_words = lda.show_topic(topicid=tid, topn=num_topic_words)
        topic_words = [word for word, _ in topic_words]
        return topic_words
    except Exception as e:
        print(f"Error in getting top topic words: {e}")
        return []

def get_topic_tokens(tokenizer: AutoTokenizer, lda: LdaModel, tid: int,
                     num_topic_words: Optional[int] = None,
                     config: Optional[ExperimentConfig] = None) -> List[int]:
    """
    Retrieves token IDs for the top words associated with a given topic from an LDA model.

    :param tokenizer: The tokenizer used to convert words to token IDs.
    :param lda: The trained LDA model.
    :param topic_id: The ID of the topic for which to retrieve top words.
    :param num_topic_words: The number of top words to retrieve for the specified topic, if any,
                            otherwise the default number of topic words is used.    
    :param config: The configuration object. Required if num_topic_words is not provided.
    :return: A list of unique token IDs corresponding to the top words of the specified topic.
    """
    if num_topic_words is None and config is None:
        logging.error("Either num_topic_words or config is required to get top topic tokens.")
        return []
    num_topic_words = config.NUM_TOPIC_WORDS if num_topic_words is None else num_topic_words
    topic_words = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
    special_token_ids = set(tokenizer.all_special_ids)
    token_ids_set = set()
    text_processor = TextProcessor()

    for word in topic_words:
        word_variations = text_processor.get_word_variations(word=word)
        token_ids = tokenizer(word_variations)['input_ids']
        token_ids = set([token for sublist in token_ids for token in sublist])
        token_ids = [token_id for token_id in token_ids if token_id not in special_token_ids]
        token_ids_set.update(token_ids)
    return list(token_ids_set)
# pylint: enable=logging-fstring-interpolation
