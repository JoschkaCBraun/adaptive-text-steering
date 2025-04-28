'''
evaluate_sentiment_of_text.py

Given a individual text, I want to evaluate the sentiment of the text.
'''

# Standard library imports
import os
import sys
import logging

# Third-party imports
from transformers import pipeline

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
# pylint: disable=wrong-import-position
from config.score_and_plot_config import ScoreAndPlotConfig 
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_language_detection_model(config: ScoreAndPlotConfig):
    '''Load the language detection model.'''
    return pipeline("text-classification", model=config.language_detection_model_name,
                    device=config.device)

def detect_language(text: str, lang_model) -> str:
    '''Detect the language of the given text.'''
    result = lang_model(text[:512])[0]  # Limit input to 512 tokens
    return result['label']
