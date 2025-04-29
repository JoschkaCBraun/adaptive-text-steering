'''
sentiment_scorer.py

Given an individual text, evaluate the sentiment using both a transformer model
and the VADER rule-based system, returning normalized scores.
'''

# Standard library imports
import sys
import logging

# Third-party imports
from transformers import pipeline as transformer_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.score_and_plot_config import ScoreAndPlotConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading Helper Functions ---

def load_transformer_sentiment_model(config: ScoreAndPlotConfig):
    '''Load the transformer-based sentiment analysis model.'''
    try:
        # Use 'text-classification' which is an alias for 'sentiment-analysis'
        sentiment_pipeline = transformer_pipeline(
            "text-classification",
            model=config.sentiment_model_name,
            device=config.device,
            truncation=True, # Ensure truncation happens if text exceeds model max length
            # max_length=config.max_tokens_for_sentiment_classification # Pass max length to pipeline
        )
        logger.info("Transformer sentiment model loaded successfully.")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"Failed to load transformer model {config.sentiment_model_name}: {e}", exc_info=True)
        raise # Re-raise the exception as loading failure is critical

def load_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Load the VADER sentiment intensity analyzer."""
    if SentimentIntensityAnalyzer is None:
         logger.error("vaderSentiment library not available. Cannot load analyzer.")
         raise ImportError("vaderSentiment library is required for this function.")

    logger.info("Loading VADER sentiment analyzer.")
    try:
        analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analyzer loaded successfully.")
        return analyzer
    except Exception as e:
        logger.error(f"Failed to load VADER analyzer: {e}", exc_info=True)
        raise # Re-raise critical loading failure

# --- Sentiment Scorer Class ---

class SentimentScorer:
    '''
    A class to load sentiment models once and perform analysis multiple times.

    Attributes:
        config (ScoreAndPlotConfig): Configuration object used for model loading.
        transformer_pipeline: The loaded Hugging Face Transformer pipeline.
        vader_analyzer (SentimentIntensityAnalyzer): The loaded VADER analyzer.
    '''
    def __init__(self, config: ScoreAndPlotConfig):
        '''
        Initializes the SentimentAnalyzer by loading the necessary models.

        Args:
            config (ScoreAndPlotConfig): Configuration object specifying model names,
                                         device, and other parameters.

        Raises:
            ImportError: If required libraries (transformers, vaderSentiment) are not installed.
            Exception: If model loading fails for other reasons.
        '''
        self.config = config
        logger.info("Initializing SentimentAnalyzer...")

        # Load models and store them as instance attributes
        # Error handling happens within the loading functions
        self.transformer_pipeline = load_transformer_sentiment_model(self.config)
        self.vader_analyzer = load_vader_analyzer()

        logger.info("SentimentAnalyzer initialized successfully.")

    def _analyze_transformer(self, text: str) -> int:
        '''Internal method for transformer analysis using the loaded pipeline.'''
        # Pipeline handles truncation based on initialization settings
        result = self.transformer_pipeline(text)[0] # Process the text
        label = result['label']

        # Extract stars - specific to models like nlptown/* returning "X stars"
        try:
            stars = int(label.split()[0])
            if not 1 <= stars <= 5:
                raise ValueError(f"Parsed star rating {stars} is outside the expected 1-5 range.")
            return stars
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not parse stars from transformer label '{label}' for text: '{text[:50]}...'. Error: {e}. Returning neutral (3 stars).")
            # Return a neutral default if label format is unexpected
            return 3 # Assuming 3 stars is neutral

    def _analyze_vader(self, text: str) -> dict:
        '''Internal method for VADER analysis using the loaded analyzer.'''
        return self.vader_analyzer.polarity_scores(text)

    def get_normalized_scores(self, text: str) -> tuple[float, float]:
        '''
        Analyzes the sentiment of a single text using the pre-loaded models
        and returns normalized scores.

        Args:
            text (str): The input text to analyze.

        Returns:
            tuple[float, float]: A tuple containing:
                                 (normalized_transformer_score, normalized_vader_score).
                                 Scores range from -1.0 (very negative) to 1.0 (very positive).
                                 Returns (0.0, 0.0) if analysis fails for the text.
        '''
        # Truncate text based on the max length defined in the config used during loading
        # Note: Transformer pipeline already handles truncation if text is longer.
        # We truncate here mainly for consistency and potentially VADER if needed,
        # although VADER doesn't have a strict token limit like transformers.
        truncated_text = text[:self.config.max_tokens_for_sentiment_classification]

        try:
            # 1. Analyze with Transformer
            stars = self._analyze_transformer(truncated_text)
            # Normalize stars (1-5) to score (-1.0 to 1.0)
            # (1 star -> -1.0), (2 stars -> -0.5), (3 stars -> 0.0), (4 stars -> 0.5), (5 stars -> 1.0)
            norm_transformer_score = (stars - 3) / 2.0

            # 2. Analyze with VADER
            vader_scores = self._analyze_vader(truncated_text)
            # VADER's 'compound' score is already normalized between -1 and 1
            norm_vader_score = vader_scores['compound']

            return {'transformer': norm_transformer_score, 'vader': norm_vader_score}

        except Exception as e:
            # Log error and return default (0.0, 0.0) as requested for analysis failures
            logger.error(f"Failed to analyze sentiment for text: '{truncated_text[:50]}...'. Error: {e}", exc_info=True)
            return {'transformer': 0.0, 'vader': 0.0} # Return neutral scores on error

# --- Main Execution Example ---

def main():
    '''Demonstrates creating a SentimentAnalyzer instance and reusing it.'''
    logger.info("Starting sentiment analysis script (Class-Based Approach).")

    # --- Configuration ---
    # If ScoreAndPlotConfig was imported successfully, use it. Otherwise, the placeholder is used.
    config = ScoreAndPlotConfig()
    logger.info(f"Using Configuration: Model={config.sentiment_model_name}, Device={config.device}, MaxTokens={config.max_tokens_for_sentiment_classification}")

    # --- Create Analyzer Instance (Loads Models ONCE) ---
    logger.info("--- Initializing Analyzer ---")
    try:
        # Create one instance of the analyzer. Models are loaded here.
        analyzer = SentimentScorer(config)
        logger.info("--- Analyzer Initialized Successfully ---")
    except (ImportError, Exception) as e:
        logger.error(f"Fatal error during analyzer initialization: {e}", exc_info=True)
        logger.error("Please ensure 'transformers', 'torch' (or 'tensorflow'/'jax'), and 'vaderSentiment' are installed.")
        sys.exit(1) # Exit if models can't load or prerequisites are missing

    # --- Example Usage: Analyze Multiple Texts ---
    logger.info("--- Running Example Usage ---")
    texts_to_analyze = [
        "This is a wonderfully fantastic product! I love it.", # Strongly positive
        "It's an okay product, does the job but nothing special.", # Neutral / Mid
        "This is terrible, absolutely horrible, a complete waste of money.", # Strongly negative
        "I am not happy with this purchase.", # Negative (negation)
        "I'm not entirely displeased, but it could be better.", # Mixed / Neutral-ish
        "The customer service was surprisingly helpful and very efficient.", # Positive
        "Despite the initial problems, the support team resolved it quickly.", # Mixed -> Positive leaning?
        "This is NOT bad at all.", # Positive (double negative emphasis)
        "What a piece of junk!", # Negative (colloquial)
        "The weather today is average.", # Neutral statement
        "😊 This makes me happy!", # Emoji positive
        "I hate waiting in long lines 😠", # Emoji negative
        "Ce produit est incroyable et fonctionne parfaitement.", # French Positive
        "Das ist eine absolute Katastrophe und Geldverschwendung.", # German Negative
    ]

    results = {}
    # Loop through the texts and call the analysis method on the *same* analyzer instance
    for i, text in enumerate(texts_to_analyze):
        # Reuse the analyzer instance; models are NOT reloaded
        norm_trans, norm_vader = analyzer.get_normalized_scores(text)

        results[i] = {'text': text, 'transformer': norm_trans, 'vader': norm_vader}
        print(f"\nAnalyzed Text {i+1}: '{text}'")
        print(f"  Normalized Scores -> Transformer: {norm_trans:.4f}, VADER: {norm_vader:.4f}")

    logger.info("--- Example Usage Finished ---")
    logger.info("Sentiment analysis script finished.")


if __name__ == "__main__":
    main()