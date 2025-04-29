'''
toxicity_scorer.py

Given an individual text, evaluate the toxicity using two different
transformer models, returning normalized scores (0=non-toxic, 1=toxic).
'''

# Standard library imports
import sys
import logging

from transformers import pipeline as transformer_pipeline

from config.score_and_plot_config import ScoreAndPlotConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading Helper Functions ---

def load_toxic_bert_pipeline(config: ScoreAndPlotConfig):
    '''Load the unitary/toxic-bert toxicity classification model pipeline.'''
    model_name = "unitary/toxic-bert"
    logger.info(f"Loading toxicity model pipeline: {model_name}")
    try:
        # Use 'text-classification' task
        toxicity_pipeline = transformer_pipeline(
            "text-classification",
            model=model_name,
            device=config.device,
            truncation=True, # Ensure truncation happens if text exceeds model max length
            max_length=config.max_tokens_for_toxicity_classification, # Pass max length to pipeline
            # Request scores for all classes to reliably get the 'toxic' score
            return_all_scores=True
        )
        logger.info(f"{model_name} pipeline loaded successfully.")
        return toxicity_pipeline
    except Exception as e:
        logger.error(f"Failed to load transformer model {model_name}: {e}", exc_info=True)
        raise # Re-raise the exception as loading failure is critical

def load_roberta_toxicity_pipeline(config: ScoreAndPlotConfig):
    '''Load the s-nlp/roberta_toxicity_classifier model pipeline.'''
    model_name = "s-nlp/roberta_toxicity_classifier"
    logger.info(f"Loading toxicity model pipeline: {model_name}")
    try:
        # Use 'text-classification' task
        toxicity_pipeline = transformer_pipeline(
            "text-classification",
            model=model_name,
            device=config.device,
            truncation=True, # Ensure truncation happens if text exceeds model max length
            max_length=config.max_tokens_for_toxicity_classification, # Pass max length to pipeline
            # Request scores for all classes to reliably get the 'toxic' score
            return_all_scores=True
        )
        logger.info(f"{model_name} pipeline loaded successfully.")
        return toxicity_pipeline
    except Exception as e:
        logger.error(f"Failed to load transformer model {model_name}: {e}", exc_info=True)
        raise # Re-raise critical loading failure

# --- Toxicity Scorer Class ---

class ToxicityScorer:
    '''
    A class to load toxicity models once and perform analysis multiple times.

    Attributes:
        config (ScoreAndPlotConfig): Configuration object used for model loading.
        toxic_bert_pipeline: The loaded Hugging Face Transformer pipeline for unitary/toxic-bert.
        roberta_toxicity_pipeline: The loaded Hugging Face Transformer pipeline for s-nlp/roberta_toxicity_classifier.
    '''
    def __init__(self, config: ScoreAndPlotConfig):
        '''
        Initializes the ToxicityScorer by loading the necessary models.

        Args:
            config (ScoreAndPlotConfig): Configuration object specifying device,
                                         max tokens, and potentially other parameters.

        Raises:
            Exception: If model loading fails.
        '''
        self.config = config
        logger.info("Initializing ToxicityScorer...")

        # Load models and store them as instance attributes
        # Error handling happens within the loading functions
        self.toxic_bert_pipeline = load_toxic_bert_pipeline(self.config)
        self.roberta_toxicity_pipeline = load_roberta_toxicity_pipeline(self.config)

        logger.info("ToxicityScorer initialized successfully.")

    def _get_score_for_label(self, results: list, target_label: str) -> float:
        """
        Helper to extract the score for a specific label from pipeline results.

        Args:
            results (list): The list of dictionaries returned by the pipeline
                            (e.g., [{'label': 'non-toxic', 'score': 0.1}, {'label': 'toxic', 'score': 0.9}]).
            target_label (str): The label whose score we want to extract (e.g., 'toxic').

        Returns:
            float: The score for the target label, or 0.0 if the label is not found.
        """
        for item in results:
            if item['label'].lower() == target_label.lower(): # Case-insensitive comparison
                return item['score']
        logger.warning(f"Target label '{target_label}' not found in results: {results}. Returning score 0.0.")
        return 0.0

    def _analyze_toxic_bert(self, text: str) -> tuple[float, float]:
        '''Internal method for toxicity analysis using the loaded toxic-bert pipeline.'''
        try:
            # Pipeline handles truncation based on initialization settings
            # Results is a list containing a list of dictionaries for each class
            # e.g. [[{'label': 'toxic', 'score': 0.9...}, {'label': 'severe_toxic', 'score': ...}]]
            result_list = self.toxic_bert_pipeline(text)[0] # Process the text, get results for the first item

            # unitary/toxic-bert has multiple toxicity labels.
            # For a single toxicity score, we usually check the primary 'toxic' label.
            # Let's explicitly look for the 'toxic' score.
            toxic_score = self._get_score_for_label(result_list, 'toxic')
            severe_toxic_score = self._get_score_for_label(result_list, 'severe_toxic')
            return toxic_score, severe_toxic_score

        except Exception as e:
            logger.error(f"Error during toxic-bert analysis for text: '{text[:50]}...'. Error: {e}", exc_info=True)
            return 0.0, 0.0 # Return neutral (non-toxic) score on error


    def _analyze_roberta_toxicity(self, text: str) -> float:
        '''Internal method for toxicity analysis using the loaded roberta-toxicity pipeline.'''
        try:
            # Pipeline handles truncation based on initialization settings
            # Results is a list containing a list of dictionaries for each class
            # e.g., [[{'label': 'non-toxic', 'score': 0.1}, {'label': 'toxic', 'score': 0.9}]]
            result_list = self.roberta_toxicity_pipeline(text)[0] # Process the text

            # Extract the score for the 'toxic' label
            toxic_score = self._get_score_for_label(result_list, 'toxic')
            return toxic_score

        except Exception as e:
            logger.error(f"Error during roberta-toxicity analysis for text: '{text[:50]}...'. Error: {e}", exc_info=True)
            return 0.0 # Return neutral (non-toxic) score on error


    def get_toxicity_scores(self, text: str) -> dict[str, float]:
        '''
        Analyzes the toxicity of a single text using the pre-loaded models
        and returns scores.

        Args:
            text (str): The input text to analyze.

        Returns:
            dict[str, float]: A dictionary containing:
                              {'toxic_bert': toxic_bert_score, 'roberta_toxicity': roberta_toxicity_score}.
                              Scores range from 0.0 (non-toxic) to 1.0 (toxic).
                              Returns {'toxic_bert': 0.0, 'roberta_toxicity': 0.0} if analysis fails.
        '''
        # Truncate text manually only if needed for non-pipeline steps,
        # but the pipeline handles truncation based on its `max_length` setting.
        # We can keep this for consistency if desired, but it's redundant for these pipelines.
        # truncated_text = text[:self.config.max_tokens_for_toxicity_classification]
        # Using original text as pipeline handles truncation.
        if not text or not isinstance(text, str) or text.isspace():
             logger.warning(f"Received empty or invalid text input. Returning zero scores.")
             return {'toxic_bert': 0.0, 'severe_toxic_bert': 0.0, 'roberta_toxicity': 0.0}


        try:
            # 1. Analyze with unitary/toxic-bert
            toxic_bert_score, severe_toxic_bert_score = self._analyze_toxic_bert(text)

            # 2. Analyze with s-nlp/roberta_toxicity_classifier
            roberta_toxicity_score = self._analyze_roberta_toxicity(text)

            return {'toxic_bert': toxic_bert_score, 'severe_toxic_bert': severe_toxic_bert_score, 'roberta_toxicity': roberta_toxicity_score}

        except Exception as e:
            # Log error and return default (0.0) as requested for analysis failures
            logger.error(f"Failed to analyze toxicity for text: '{text[:50]}...'. Error: {e}", exc_info=True)
            return {'toxic_bert': 0.0, 'severe_toxic_bert': 0.0, 'roberta_toxicity': 0.0} # Return non-toxic scores on error

# --- Main Execution Example ---

def main():
    '''Demonstrates creating a ToxicityScorer instance and reusing it.'''
    logger.info("Starting toxicity analysis script.")

    # --- Configuration ---
    # Instantiate the configuration object
    config = ScoreAndPlotConfig()
    logger.info(f"Using Configuration: Device={config.device}, MaxTokens={config.max_tokens_for_toxicity_classification}")

    # --- Create Analyzer Instance (Loads Models ONCE) ---
    logger.info("--- Initializing Analyzer ---")
    try:
        # Create one instance of the analyzer. Models are loaded here.
        analyzer = ToxicityScorer(config)
        logger.info("--- Analyzer Initialized Successfully ---")
    except Exception as e:
        logger.error(f"Fatal error during analyzer initialization: {e}", exc_info=True)
        logger.error("Please ensure 'transformers' and 'torch' (or 'tensorflow'/'jax') are installed.")
        sys.exit(1) # Exit if models can't load

    # --- Example Usage: Analyze Multiple Texts ---
    logger.info("--- Running Example Usage ---")
    texts_to_analyze = [
        "This is a perfectly nice and friendly comment.", # Non-toxic
        "Have a great day everyone!", # Non-toxic
        "You hideous jewish cunt. I'm going to kill you.", # Toxic (Identity Attack/Hate)
        "You are such an idiot, I can't believe you wrote that.", # Toxic (Insult)
        "I'm going to find you and hurt you.", # Toxic (Threat)
        "This movie was terrible, complete garbage.", # Negative sentiment, likely non-toxic
        "What a moron!", # Toxic (Insult)
        "You stupid fucking cunt!", # Toxic (Identity Attack/Hate)
        "Some people are just stupid.", # Borderline/Toxic (Insult)
        "Sunshine and rainbows, motherfucker.", # Non-toxic
        "", # Empty string
        "   ", # Whitespace only
        "This is some neutral text about technical specifications.", # Neutral, non-toxic
        "Je déteste ce genre de comportement stupide.", # French (potentially toxic depending on model capability)
        "Das ist absolut widerlich und inakzeptabel.", # German (potentially toxic depending on model capability)

    ]

    results = {}
    # Loop through the texts and call the analysis method on the *same* analyzer instance
    for i, text in enumerate(texts_to_analyze):
        # Reuse the analyzer instance; models are NOT reloaded
        scores = analyzer.get_toxicity_scores(text)

        results[i] = {'text': text, **scores} # Combine text with scores
        print(f"\nAnalyzed Text {i+1}: '{text}'")
        print(f"  Toxicity Scores -> Toxic-BERT: {scores['toxic_bert']:.4f}, Severe Toxic-BERT: {scores['severe_toxic_bert']:.4f}, RoBERTa-Toxicity: {scores['roberta_toxicity']:.4f}")

    logger.info("--- Example Usage Finished ---")
    logger.info("Toxicity analysis script finished.")


if __name__ == "__main__":
    main()