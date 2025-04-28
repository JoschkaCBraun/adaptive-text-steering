'''
scorer.py

A unified scorer class that orchestrates intrinsic quality, extrinsic quality, topic,
and sentiment scoring for text.

It initializes individual scorer components once (Composition) to efficiently
reuse loaded models and resources.
'''

# Standard library imports
import logging
from typing import Dict, Optional, Any

from config.score_and_plot_config import ScoreAndPlotConfig

from src.evaluation.sentiment_scorer import SentimentScorer
from src.evaluation.topic_scorer import TopicScorer
from src.evaluation.intrinsic_quality_scorer import IntrinsicQualityScorer
from src.evaluation.extrinsic_quality_scorer import ExtrinsicQualityScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Scorer:
    """
    A unified scorer class that orchestrates intrinsic quality, extrinsic quality, topic,
    and sentiment scoring for text.

    It initializes individual scorer components once (Composition) to efficiently
    reuse loaded models and resources.
    """
    def __init__(self):
        """
        Initializes the main Scorer by loading all necessary sub-scorer components.

        Args:
            config (ScoreAndPlotConfig): The configuration object containing settings
                                         for all underlying scorers.

        Raises:
            Exception: Propagates exceptions raised during the initialization
                       of any sub-scorer (e.g., model loading failures).
        """
        logger.info("Initializing main Scorer...")
        self.config = ScoreAndPlotConfig()

        try:
            logger.info("Initializing SentimentScorer...")
            self.sentiment_scorer = SentimentScorer(self.config)
            logger.info("SentimentScorer initialized.")
            logger.info("Initializing TopicScorer...")
            self.topic_scorer = TopicScorer(self.config)
            logger.info("TopicScorer initialized.")
            logger.info("Initializing IntrinsicQualityScorer...")
            self.intrinsic_scorer = IntrinsicQualityScorer(self.config)
            logger.info("IntrinsicQualityScorer initialized.")
            logger.info("Initializing ExtrinsicQualityScorer...")
            self.extrinsic_scorer = ExtrinsicQualityScorer(self.config)
            logger.info("ExtrinsicQualityScorer initialized.")

            logger.info("Main Scorer initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize main Scorer due to error in sub-component: {e}", exc_info=True)
            # Re-raise the exception to signal critical failure
            raise

    def score_individual_text(self,
                              text: str,
                              tid1: Optional[int] = None,
                              tid2: Optional[int] = None,
                              reference_text1: Optional[str] = None,
                              reference_text2: Optional[str] = None,
                              topic_method: str = None,
                              distinct_n: int = 2 # Default n for distinct n-grams
                             ) -> Dict[str, Any]:
        """
        Scores a single text using the configured and initialized scorers.

        Args:
            text (str): The input text to score.
            tid1 (Optional[int]): The first topic ID to score against. If None,
                                  topic scoring for tid1 is skipped. Defaults to None.
            tid2 (Optional[int]): The second topic ID to score against. If None,
                                  topic scoring for tid2 is skipped. Defaults to None.
            reference_text1 (Optional[str]): The first reference text for extrinsic
                                             comparison. If None, extrinsic scoring
                                             against ref1 is skipped. Defaults to None.
            reference_text2 (Optional[str]): The second reference text for extrinsic
                                             comparison. If None, extrinsic scoring
                                             against ref2 is skipped. Defaults to None.
            topic_method (str): The method for topic scoring (e.g., 'dict', 'tokenize',
                                'stem', 'lemmatize'). This method will be used for
                                both tid1 and tid2 if provided. Defaults to 'lemmatize'.
            distinct_n (int): The 'n' value for the Distinct-N calculation within
                              intrinsic quality scoring. Defaults to 2.

        Returns:
            Dict[str, Any]: A dictionary containing the input text, identifiers used,
                            and nested dictionaries for sentiment, intrinsic, topic,
                            and extrinsic scores. Scores that failed or were skipped
                            might be absent or have None/default error values.
        """
        if not isinstance(text, str):
             logger.warning("Input 'text' is not a string. Skipping scoring.")
             return {"text": text, "error": "Input text must be a string."}
             
        logger.debug(f"Scoring text (first 50 chars): '{text[:50]}...'")
        scores: Dict[str, Any] = {"text": text}
        if tid1 is not None:
            scores["tid1_used"] = tid1
        if tid2 is not None:
            scores["tid2_used"] = tid2
        if reference_text1 is not None:
            scores["reference_text1_used"] = reference_text1
            if not isinstance(reference_text1, str):
                logger.warning("Input 'reference_text1' is not a string. Skipping extrinsic scoring.")
                reference_text1 = None
        if reference_text2 is not None:
            scores["reference_text2_used"] = reference_text2
            if not isinstance(reference_text2, str):
                logger.warning("Input 'reference_text2' is not a string. Skipping extrinsic scoring.")
                reference_text2 = None
        if distinct_n is not None:
            scores["distinct_n_used"] = distinct_n
        if topic_method is not None:
            scores["topic_method_used"] = topic_method

        scores['sentiment_scores'] = {}
        scores['intrinsic_scores'] = {}
        if tid1 is not None:
            scores['topic_scores'] = {}
        if reference_text1 is not None:
            scores['extrinsic_scores'] = {}

        # 1. Sentiment Scoring (Always runs if text is valid)
        try:
            sentiment_scores_dict = self.sentiment_scorer.get_normalized_scores(text)
            scores['sentiment_scores'] = sentiment_scores_dict
        except Exception as e:
            logger.error(f"Error during sentiment scoring: {e}", exc_info=False)

        # 2. Intrinsic Quality Scoring (Always runs if text is valid)
        try:
            intrinsic_scores_dict = self.intrinsic_scorer.get_intrinsic_scores(text, n_value=distinct_n)
            scores['intrinsic_scores'].update(intrinsic_scores_dict)
        except Exception as e:
            logger.error(f"Error during intrinsic scoring: {e}", exc_info=False)

        # 3. Topic Scoring (Conditional on tid)
        if tid1 is not None:
            try:
                topic_scores_tid1_dict = self.topic_scorer.get_topic_score(text=text, tid=tid1, method=topic_method)
                scores['topic_scores']['tid1'] = topic_scores_tid1_dict
            except Exception as e:
                logger.error(f"Unexpected error during topic scoring for tid {tid1}: {e}", exc_info=False)
        else:
            logger.debug("Topic ID (tid) not provided, skipping topic scoring.")

        if tid2 is not None:
            try:
                topic_scores_tid2_dict = self.topic_scorer.get_topic_score(text=text, tid=tid2, method=topic_method)
                scores['topic_scores']['tid2'] = topic_scores_tid2_dict
            except Exception as e:
                logger.error(f"Unexpected error during topic scoring for tid {tid2}: {e}", exc_info=False)
        else:
            logger.debug("Topic ID (tid) not provided, skipping topic scoring.")

        # 4. Extrinsic Quality Scoring (Conditional on reference_text)
        if reference_text1 is not None:
            try:
                extrinsic_scores1_dict = self.extrinsic_scorer.get_similarity_scores(
                    generated_text=text,
                    reference_text=reference_text1
                )
                scores['extrinsic_scores']['reference_text1'] = extrinsic_scores1_dict
            except Exception as e:
                logger.error(f"Error during extrinsic scoring: {e}", exc_info=False)

        if reference_text2 is not None:
            try:
                extrinsic_scores2_dict = self.extrinsic_scorer.get_similarity_scores(
                    generated_text=text,
                    reference_text=reference_text2
                )
                scores['extrinsic_scores']['reference_text2'] = extrinsic_scores2_dict
            except Exception as e:
                logger.error(f"Error during extrinsic scoring: {e}", exc_info=False)

        logger.debug(f"Finished scoring text (first 50 chars): '{text[:50]}...'")
        return scores