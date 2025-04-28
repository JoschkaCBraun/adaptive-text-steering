"""
topic_scorer.py
This script defines the TopicScorer class, which calculates the topic focus
score for a single given text and topic ID using different methods.
It loads necessary resources like LDA models, dictionaries, and tokenizers once.
"""

# Standard library imports
import logging
import sys
from typing import Set, List, Tuple, Optional, Dict

# Third-party imports
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Local imports
from src.utils.lda_utils import load_lda, load_dictionary, get_topic_tokens     
from src.utils.text_processing_utils import TextProcessor
from config.score_and_plot_config import ScoreAndPlotConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation

class TopicScorer:
    """
    Calculates topic scores for individual texts based on a pre-loaded LDA model.

    Loads LDA model, dictionary, tokenizer, and text processor during initialization.
    Provides methods to score a text against a topic ID using 'dict', 'tokenize',
    'stem', or 'lemmatize' methods.
    """

    VALID_METHODS: List[str] = ['dict', 'tokenize', 'stem', 'lemmatize']
    VALID_TID_RANGE: Tuple[int, int] = (0, 250)

    def __init__(self, config: ScoreAndPlotConfig):
        """
        Initializes the TopicScorer by loading necessary models and resources.

        Args:
            config (ScoreAndPlotConfig): Configuration object containing settings like
                                         tokenizer_name_for_topic_scorer and num_topic_words.

        Raises:
            AttributeError: If the config object is missing required attributes.
            FileNotFoundError: If model or dictionary files specified by utils are not found.
            Exception: If resource loading fails for other reasons.
        """
        logger.info("Initializing TopicScorer...")
        self.config = config

        # Validate required config attributes
        if not hasattr(self.config, 'num_topic_words'):
            raise AttributeError("Configuration object must have attribute 'num_topic_words'.")
        if not hasattr(self.config, 'tokenizer_name_for_topic_scorer'):
            # Example attribute name, adjust if different in your actual ScoreAndPlotConfig
            raise AttributeError("Configuration object must have attribute 'tokenizer_name_for_topic_scorer'.")

        # --- Load Resources ---
        try:
            logger.info("Loading LDA model...")
            self.lda: LdaModel = load_lda()
            if self.lda is None:
                raise FileNotFoundError("Failed to load LDA model (load_lda returned None).")
            logger.info("LDA model loaded successfully.")

            logger.info("Loading Gensim dictionary...")
            self.dictionary: Dictionary = load_dictionary()
            if self.dictionary is None:
                raise FileNotFoundError("Failed to load Gensim dictionary (load_dictionary returned None).")
            logger.info("Gensim dictionary loaded successfully.")

            logger.info(f"Loading tokenizer: {self.config.tokenizer_name_for_topic_scorer}...")
            self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.config.tokenizer_name_for_topic_scorer)
            logger.info("Tokenizer loaded successfully.")

            logger.info("Initializing TextProcessor...")
            self.text_processor: TextProcessor = TextProcessor()
            logger.info("TextProcessor initialized successfully.")

        except FileNotFoundError as e:
            logger.error(f"Initialization failed: Resource file not found. {e}", exc_info=True)
            raise # Re-raise critical loading failure
        except AttributeError as e:
             logger.error(f"Initialization failed: Configuration error. {e}", exc_info=True)
             raise # Re-raise config error
        except Exception as e:
            logger.error(f"Initialization failed due to an unexpected error during resource loading: {e}", exc_info=True)
            raise # Re-raise any other critical loading failure

        self.num_topic_words = self.config.num_topic_words
        logger.info("TopicScorer initialized successfully.")


    def get_topic_score(self, text: str, tid: int, method: Optional[str] = None) -> Dict[str, float]:
        """
        Calculates the topic score for the given text and topic ID using the specified method.

        Args:
            text (str): The input text to score.
            tid (int): The target topic ID.
            method (str): The scoring method to use. Must be one of
                          'dict', 'tokenize', 'stem', or 'lemmatize'.

        Returns:
            float: The calculated topic score.

        Raises:
            ValueError: If the provided method is invalid or if a calculation error occurs
                        (e.g., division by zero).
            Exception: For other unexpected errors during calculation.
        """
        logger.debug(f"Calculating topic score for tid {tid} using method '{method}'")
        method = method.lower() if method else None
        methods_to_run: List[str] = [method] if method else self.VALID_METHODS

        if not isinstance(tid, int):
            raise ValueError(f"Invalid topic ID: {tid}. Must be an integer.")
        if tid not in range(*self.VALID_TID_RANGE):
            raise ValueError(f"Invalid topic ID: {tid}. Must be between {self.VALID_TID_RANGE[0]} and {self.VALID_TID_RANGE[1]}.")
        if method is not None and method not in self.VALID_METHODS:
            raise ValueError(f"Invalid scoring method: '{method}'. Choose from {self.VALID_METHODS}.")
        if not text:
             logger.warning("Input text is empty. Returning score 0.0.")
             return {m: 0.0 for m in methods_to_run}
        
        results: Dict[str, float] = {}

        try:
            for method in methods_to_run:
                score: float = 0.0
                if method == 'dict':
                    score = self._calculate_dict_score(text, tid)
                elif method == 'tokenize':
                    score = self._calculate_tokenize_score(text, tid)
                elif method == 'stem':
                    score = self._calculate_stem_lem_score(text, tid, 'stem')
                elif method == 'lemmatize':
                    score = self._calculate_stem_lem_score(text, tid, 'lemmatize')
                results[method] = score
        except ZeroDivisionError as e:
            logger.error(f"Calculation error for method '{method}', tid {tid}: Division by zero. Text: '{text[:50]}...'")
            # Set score to 0.0 instead of raising error
            results[method] = 0.0
        except Exception as e:
            logger.error(f"Unexpected error during topic score calculation for method '{method}', tid {tid}. Error: {e}", exc_info=True)
            results[method] = 0.0  # Return 0.0 on error

        return results

    def _calculate_dict_score(self, text: str, tid: int) -> float:
        """Calculates topic score using the Gensim dictionary and LDA model."""
        logger.debug(f"Calculating 'dict' score for tid {tid}")
        # Prepare text for dictionary
        processed_text = text.lower().split()
        if not processed_text:
            return 0.0

        # Convert text to bag-of-words representation
        vec_bow = self.dictionary.doc2bow(processed_text)

        # Get topic distribution for the document
        # Ensure minimum_probability=0 to get all topics, even with low probability
        document_topics = dict(self.lda.get_document_topics(vec_bow, minimum_probability=0))

        # Get the score for the target topic ID, default to 0.0 if not present
        score = document_topics.get(tid, 0.0)
        logger.debug(f"'dict' score for tid {tid}: {score}")
        return score


    def _calculate_tokenize_score(self, text: str, tid: int) -> float:
        """Calculates topic score based on tokenizer tokens."""
        logger.debug(f"Calculating 'tokenize' score for tid {tid}")
        # Get the set of token IDs for the target topic
        topic_token_ids: Set[int] = set(get_topic_tokens(
            tokenizer=self.tokenizer,
            lda=self.lda,
            tid=tid,
            num_topic_words=self.num_topic_words
        ))

        if not topic_token_ids:
            logger.warning(f"No topic tokens found for tid {tid}. Returning score 0.0.")
            return 0.0

        # Tokenize the input text word by word and flatten
        words = text.lower().split()
        tokenized_text_ids = []
        for word in words:
            # Use add_special_tokens=False if you don't want BOS/EOS etc. within word tokens
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            tokenized_text_ids.extend(word_tokens)

        if not tokenized_text_ids:
            logger.warning(f"Text resulted in zero tokens after tokenization: '{text[:50]}...'")
            return 0.0  # Return 0 instead of raising error

        # Count matching tokens
        matching_tokens_count = sum(1 for token_id in tokenized_text_ids if token_id in topic_token_ids)

        # Calculate normalized score
        total_tokens = len(tokenized_text_ids)
        normalized_score = float(matching_tokens_count) / float(total_tokens)

        logger.debug(f"'tokenize' score for tid {tid}: {normalized_score} ({matching_tokens_count}/{total_tokens})")
        return normalized_score


    def _calculate_stem_lem_score(self, text: str, tid: int, process_method: str) -> float:
        """Calculates topic score based on stemmed or lemmatized words."""
        logger.debug(f"Calculating '{process_method}' score for tid {tid}")
        # Process the text using the specified method (stem or lemmatize)
        # This removes stopwords and punctuation as per TextProcessor logic
        processed_words: Set[str] = self.text_processor.process_text(text, process_method)

        if not processed_words:
            logger.warning(f"Text resulted in zero processed words using method '{process_method}': '{text[:50]}...'")
            return 0.0  # Return 0 instead of raising error

        # Get the top N words and their weights for the target topic
        try:
            # Note: show_topic returns list of (word, weight) tuples
            topic_words_with_weights = dict(self.lda.show_topic(tid, topn=self.num_topic_words))
        except IndexError:
             logger.error(f"Topic ID {tid} is invalid for the loaded LDA model.")
             return 0.0  # Return 0 instead of raising error

        if not topic_words_with_weights:
             logger.warning(f"LDA model returned no words for tid {tid}. Returning score 0.0.")
             return 0.0

        # Calculate the total weight of the topic's top words
        total_weight = sum(topic_words_with_weights.values())

        if total_weight == 0.0:
            logger.warning(f"Total weight for top words of tid {tid} is zero.")
            return 0.0  # Return 0 instead of raising error

        # Calculate the weighted score based on matching processed words
        weighted_score = sum(topic_words_with_weights.get(word, 0.0) for word in processed_words)

        # Normalize the score
        normalized_score = weighted_score / total_weight

        logger.debug(f"'{process_method}' score for tid {tid}: {normalized_score} ({weighted_score}/{total_weight})")
        return normalized_score


# --- Main Execution Example ---

def main():
    """Demonstrates initializing and using the TopicScorer."""
    logger.info("Starting topic scoring script demonstration.")
    config = ScoreAndPlotConfig()

    # --- Initialize Scorer (Loads Models ONCE) ---
    logger.info("--- Initializing TopicScorer ---")
    try:
        scorer = TopicScorer(config)
        logger.info("--- TopicScorer Initialized Successfully ---")
    except (AttributeError, FileNotFoundError, Exception) as e:
        logger.error(f"Fatal error during scorer initialization: {e}", exc_info=True)
        sys.exit(1)

    # --- Example Texts with Different Topics ---
    example_texts = [
        # Text 1: Politics and Government
        "Guatemala's Constitutional Court barred Sandra Torres, the country's former first lady, from running for president. Torres has now exhausted her appeals after divorcing Guatemala's current president to try to circumvent a nepotism law. The court unanimously voted that Torres was not eligible because close relatives of the president are barred from running according to the Constitution. ",
        
        # Text 2: Technology and Science
        "Artificial intelligence and machine learning algorithms are transforming how we process data. Researchers continue to develop neural networks that can solve complex problems.",
        
        # Text 3: Sports
        "The basketball team won the championship after an intense overtime period. The coach's strategy paid off when the star player scored the winning points in the final seconds.",
        
        # Text 4: Health and Medicine
        "Medical researchers have developed a new vaccine that shows promising results in clinical trials. The study suggests it could prevent several variants of the virus.",
        
        # Text 5: Empty text (to test handling)
        ""
    ]
    
    example_tids = [78, 13, 42, 120, 5]  # Example topic IDs to test with
    methods_to_test = ['dict', 'tokenize', 'lemmatize', 'stem']
    
    # --- Score all example texts with all methods ---
    logger.info("\n--- Scoring Multiple Example Texts ---")
    
    for i, (text, tid) in enumerate(zip(example_texts, example_tids)):
        logger.info(f"\n--- Example {i+1} (Topic ID: {tid}) ---")
        if text:
            logger.info(f"  Text: '{text[:100]}...'")
        else:
            logger.info("  Text: [Empty string]")
        
        for method in methods_to_test:
            try:
                result = scorer.get_topic_score(text=text, tid=tid, method=method)
                print(f"  Score ({method}): {result[method]:.6f}")
            except Exception as e:
                logger.error(f"Unexpected error scoring example {i+1} with method {method}: {e}")
                print(f"  Score ({method}): ERROR - Continuing with next method")
    
    # --- Example with stopwords only ---
    logger.info("\n--- Scoring Example with only stopwords ---")
    stopwords_text = "a the is of and"
    logger.info(f"  Text: '{stopwords_text}'")
    for method in methods_to_test:
        try:
            result = scorer.get_topic_score(text=stopwords_text, tid=example_tids[0], method=method)
            print(f"  Score ({method}): {result[method]:.6f}")
        except Exception as e:
            logger.error(f"Unexpected error scoring stopwords text with method {method}: {e}")
            print(f"  Score ({method}): ERROR - Continuing with next method")

    # --- Ask user for custom text and topic ID ---
    logger.info("\n--- Topic scoring script demonstration finished. ---")
    logger.info("To perform your own scoring test, you can modify this script to input your own text and topic ID.")
    logger.info("Example: scorer.get_topic_score(text='Your custom text here', tid=42, method='dict')")


if __name__ == "__main__":
    main()