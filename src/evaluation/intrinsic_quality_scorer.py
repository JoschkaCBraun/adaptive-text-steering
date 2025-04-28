'''
intrinsic_quality_scorer.py
This script evaluates the intrinsic quality of a single generated text using metrics like
Perplexity, Word N-gram Repetition (Distinct-Word-N), and Character N-gram Repetition (Distinct-Char-N).
'''

# Standard library imports
import sys
import logging
import re
from typing import Dict, Optional

# --- Third-party imports ---
import torch
import nltk
from nltk.util import ngrams

# --- Local imports ---
from src.utils import load_model_and_tokenizer
from config.score_and_plot_config import ScoreAndPlotConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download nltk data if not present (needed for ngrams)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK punkt tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
    logger.debug("NLTK 'punkt_tab' resource found.")
except LookupError:
    logger.info("NLTK 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt_tab', quiet=True) # Download punkt_tab
    logger.info("NLTK 'punkt_tab' resource downloaded.")

# --- Intrinsic Quality Scorer Class ---

class IntrinsicQualityScorer:
    """
    Calculates intrinsic quality metrics (Perplexity, Distinct-Word-N, Distinct-Char-N)
    for generated text.

    Loads a specified language model for perplexity calculation once upon initialization.
    Uses NLTK for word tokenization and N-gram calculation.
    """
    def __init__(self,
                 config: ScoreAndPlotConfig,
                 ppl_model_alias: Optional[str] = None,
                 default_n: int = 2):
        """
        Initializes the scorer by loading the language model for perplexity.

        Args:
            config (ScoreAndPlotConfig): Configuration object. Expected to have
                                         'perplexity_model_alias' if ppl_model_alias is not provided.
            ppl_model_alias (Optional[str]): Specific model alias to use for perplexity.
                                             Overrides the one in the config if provided.
                                             Defaults to None.
            default_n (int): The default value of 'n' for Distinct-N calculation.
                             Defaults to 2.

        Raises:
            AttributeError: If neither ppl_model_alias is provided nor
                            config.perplexity_model_alias exists.
            ValueError: If the resolved perplexity_model_alias is empty.
            Exception: If model loading via load_model_and_tokenizer fails.
        """
        logger.info("Initializing IntrinsicQualityScorer...")
        self.config = config
        self.default_n = default_n

        # Determine the perplexity model alias
        model_alias_to_load = ppl_model_alias
        if model_alias_to_load is None:
            if not hasattr(config, 'perplexity_model_alias'):
                raise AttributeError("Configuration object must have 'perplexity_model_alias' or "
                                     "'ppl_model_alias' must be provided.")
            model_alias_to_load = config.perplexity_model_alias
            if not model_alias_to_load:
                 raise ValueError("Config's 'perplexity_model_alias' cannot be empty if used.")

        logger.info(f"Preparing to load perplexity model: {model_alias_to_load}")

        # Load the model, tokenizer, and device ONCE
        try:
            # We assume load_model_and_tokenizer handles device placement
            self.ppl_tokenizer, self.ppl_model, self.ppl_device = load_model_and_tokenizer(
                model_alias=model_alias_to_load
            )
            self.ppl_model.eval()
            logger.info(f"Perplexity model '{model_alias_to_load}' loaded successfully on device '{self.ppl_device}'.")
            # Store max length for truncation warnings
            try:
                self.max_length = self.ppl_model.config.max_position_embeddings
            except AttributeError:
                 logger.warning("Could not determine model's max_position_embeddings from config. Using fallback length 512.")
                 self.max_length = 512

        except Exception as e:
            logger.error(f"Failed to load perplexity model '{model_alias_to_load}': {e}", exc_info=True)
            raise

        logger.info("IntrinsicQualityScorer initialized successfully.")

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculates the perplexity of a given text using the loaded language model.

        Perplexity is calculated as exp(average negative log-likelihood).
        Lower perplexity indicates the model found the text more predictable/fluent.

        Args:
            text (str): The input text.

        Returns:
            float: The calculated perplexity score. Returns float('inf') if text is empty,
                   if the model couldn't be loaded, or if loss calculation fails.
        """
        if not hasattr(self, 'ppl_model') or not hasattr(self, 'ppl_tokenizer'):
             logger.error("Perplexity model/tokenizer not available. Cannot calculate perplexity.")
             return float('inf')

        if not text:
            logger.warning("Input text is empty. Returning perplexity as infinity.")
            return float('inf')

        try:
            encodings = self.ppl_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False # Don't pad for PPL calculation unless absolutely necessary for model
            )

            input_ids = encodings.input_ids.to(self.ppl_device)
            target_ids = input_ids.clone() # Labels are the input_ids for Causal LM PPL
            seq_len = input_ids.size(1)

            if seq_len == 0:
                logger.warning("Input text resulted in zero tokens after tokenization. Returning perplexity as infinity.")
                return float('inf')

            # Check if truncation occurred
            if encodings.get('num_truncated_tokens', 0) > 0:
                 logger.warning(f"Input text was truncated to {self.max_length} tokens for perplexity calculation.")

            with torch.no_grad():
                outputs = self.ppl_model(input_ids, labels=target_ids)
                # The loss returned is the average cross-entropy loss over the sequence
                neg_log_likelihood = outputs.loss

            if neg_log_likelihood is None or torch.isnan(neg_log_likelihood) or torch.isinf(neg_log_likelihood):
                logger.warning(f"Perplexity calculation resulted in invalid loss ({neg_log_likelihood}). Returning infinity.")
                return float('inf')

            # Calculate perplexity: PPL = exp(average NLL)
            perplexity = torch.exp(neg_log_likelihood)
            return perplexity.item()

        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}", exc_info=True)
            return float('inf')

    def _calculate_distinct_word_n(self, text: str, n: int) -> float:
        """
        Internal helper to calculate Distinct-Word-N using nltk.word_tokenize.
        """
        if not text:
            return 0.0
        # Use nltk.word_tokenize and lowercase
        tokens = nltk.word_tokenize(text.lower())
        if len(tokens) < n:
            return 0.0
        n_grams_list = list(ngrams(tokens, n))
        if not n_grams_list:
             return 0.0
        distinct_n_grams_count = len(set(n_grams_list))
        total_n_grams_count = len(n_grams_list)
        return distinct_n_grams_count / total_n_grams_count

    def _calculate_distinct_char_n(self, text: str, n: int) -> float:
        """
        Internal helper to calculate Distinct-Character-N.
        """
        if not text:
            return 0.0
        # Use lowercase string as sequence of characters
        processed_text = text.lower()
        if len(processed_text) < n:
            return 0.0
        char_n_grams_list = list(ngrams(processed_text, n))
        if not char_n_grams_list:
            return 0.0
        distinct_n_grams_count = len(set(char_n_grams_list))
        total_n_grams_count = len(char_n_grams_list)
        return distinct_n_grams_count / total_n_grams_count

    def get_intrinsic_scores(self, generated_text: str, n_value: Optional[int] = None) -> Dict[str, float]:
        """
        Calculates and returns a dictionary of intrinsic quality scores for the generated text.
        Specifically calculates Perplexity, Distinct-Word-N, and Distinct-Char-N.

        Args:
            generated_text (str): The text whose quality is to be evaluated.
            n_value (Optional[int]): Specific 'n' to use for Distinct-N. If None, uses
                                     the default 'n' set during initialization.

        Returns:
            Dict[str, float]: A dictionary containing the calculated scores:
                              {'perplexity': PPL_score,
                               f'distinct_word_{used_n}': Distinct-Word-N_score,
                               f'distinct_char_{used_n}': Distinct-Char-N_score}
                              Returns default error values (inf for PPL, 0.0 for Distinct)
                              if the input text is empty.
        """
        used_n = n_value if n_value is not None else self.default_n

        if not generated_text:
             logger.warning("Received empty generated text. Returning default scores.")
             return {'perplexity': float('inf'),
                     f'distinct_word_{used_n}': 0.0,
                     f'distinct_char_{used_n}': 0.0}

        # Calculate Perplexity
        ppl_score = self.calculate_perplexity(generated_text)

        # Calculate Distinct-Word-2
        distinct_word_n_score = self._calculate_distinct_word_n(generated_text, n=used_n)

        # Calculate Distinct-Char-2
        distinct_char_n_score = self._calculate_distinct_char_n(generated_text, n=used_n)

        scores = {
            'perplexity': ppl_score,
            f'distinct_word_{used_n}': distinct_word_n_score,
            f'distinct_char_{used_n}': distinct_char_n_score
        }
        return scores

# --- Main Execution Example ---

def main():
    """Demonstrates initializing and using the IntrinsicQualityScorer."""
    logger.info("Starting intrinsic quality evaluation script example.")

    try:
        config = ScoreAndPlotConfig()
        logger.info(f"Using Configuration: Perplexity Model Alias={config.perplexity_model_alias}")

        scorer = IntrinsicQualityScorer(config=config)

    except (AttributeError, ValueError, Exception) as e:
        logger.error(f"Fatal error during scorer initialization: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Scorer Initialized Successfully ---")

    # --- Test Cases ---
    test_texts = [
        {
            "id": "Fluent Sentence",
            "text": "The quick brown fox jumps over the lazy dog near the river bank."
        },
        {
            "id": "Repetitive Sentence (Word)",
            "text": "the cat sat on the mat the cat sat on the mat the cat sat"
        },
         {
            "id": "Repetitive Sentence (Char)",
            "text": "ababababababababababababababa"
        },
        {
            "id": "Repetitive Punctuation",
            "text": "This is........................... amazing!"
        },
        {
            "id": "Less Fluent / Uncommon",
            "text": "Colorless green ideas sleep furiously alongside virtual realities."
        },
        {
            "id": "Short Text",
            "text": "Hello world."
        },
         {
            "id": "Empty Text",
            "text": ""
        }
    ]

    logger.info("--- Running Test Cases ---")
    for case in test_texts:
        logger.info(f"\n--- Test Case: {case['id']} ---")
        logger.info(f"  Text: '{case['text']}'")

        # Get scores using the scorer instance
        intrinsic_scores = scorer.get_intrinsic_scores(generated_text=case['text'])
        print(f"  Scores:")
        for name, score in intrinsic_scores.items():
            # Format perplexity nicely, distinct metrics to 4 decimal places
            if name == 'perplexity':
                print(f"    {name}: {score:.4f}" if score != float('inf') else f"    {name}: inf")
            else:
                print(f"    {name}: {score:.4f}")

    logger.info("\n--- Test Cases Finished ---")
    logger.info("Intrinsic quality evaluation script example finished.")


if __name__ == "__main__":
    # Make sure NLTK data is available before running main
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' resource...")
        nltk.download('punkt', quiet=True)
        print("'punkt' downloaded.")
    main()