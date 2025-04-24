'''
evaluate_sentiment_of_text.py

Given a individual text, I want to evaluate the sentiment of the text.
'''

# Standard library imports
import os
import sys
import json
import logging
from typing import List, Dict, Any

# Third-party imports
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
# pylint: disable=wrong-import-position
from config.score_and_plot_config import ScoreAndPlotConfig 
from utils import load_model_and_tokenizer
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sentiment_model(config: ScoreAndPlotConfig):
    '''Load the sentiment analysis model.'''
    return pipeline("text-classification", model=config.sentiment_model_name, device=config.device)

def load_language_detection_model(config: ScoreAndPlotConfig):
    '''Load the language detection model.'''
    return pipeline("text-classification", model=config.language_detection_model_name,
                    device=config.device)

def analyze_sentiment(text: str, sentiment_pipeline) -> tuple:
    '''Analyze the sentiment of the given text.'''
    result = sentiment_pipeline(text[:512])[0]  # Limit input to 512 tokens
    label = result['label']
    score = result['score']

    # extract the number of stars from the label
    stars = int(label.split()[0])

    if stars > 5: 
        stars = (stars / 2)

    return stars, score

def detect_language(text: str, lang_model) -> str:
    '''Detect the language of the given text.'''
    result = lang_model(text[:512])[0]  # Limit input to 512 tokens
    return result['label']




def evaluate_all_reviews(reviews: List[Dict[str, Any]], config: ScoreAndPlotConfig) -> List[Dict[str, Any]]:
    '''Evaluate all reviews in the given list.'''
    sentiment_pipeline = load_sentiment_model(config)
    lang_model = load_language_detection_model(config)
    # temporary fix for loading the model and tokenizer
    hf_auth_token = config.hf_auth_token
    coherence_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                                           token=hf_auth_token)
    coherence_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="google/gemma-2b-it",
                                                        token=hf_auth_token)
    
    evaluated_reviews = []
    for review in reviews:
        evaluated_review = evaluate_review(review, sentiment_pipeline, lang_model, coherence_model, coherence_tokenizer)
        evaluated_reviews.append(evaluated_review)
        logger.info(f"Evaluated review for movie: {review['movie']} in language: {review['prompt_language']}")
    
    return evaluated_reviews

def main():
    '''Main execution flow for analyzing sentiment vectors.'''
    config = ScoreAndPlotConfig()
    # set MODEL_ALIAS to the model alias you want to evaluate
    MODEL_ALIAS = "gemma_2b"
    MOVIE_REVIEW_COUNT = 450
    NUM_TUPLES = 250

    sentiment_vectors_data_dir = os.path.join(get_data_dir(os.getcwd()), 'sentiment_vectors_data')
    
    # Create scores directory if it doesn't exist
    scores_dir = os.path.join(sentiment_vectors_data_dir, 'sentiment_vectors_scores')
    os.makedirs(scores_dir, exist_ok=True)

    # Load your JSON results
    input_file = os.path.join(sentiment_vectors_data_dir, 'sentiment_vectors_results',
                              f"results_{NUM_TUPLES}_{MODEL_ALIAS}_{MOVIE_REVIEW_COUNT}.json")
    with open(input_file, 'r') as f:
        reviews = json.load(f)
    
    evaluated_reviews = evaluate_all_reviews(reviews, config)
    
    # Save the evaluated reviews
    output_file = os.path.join(scores_dir,
                               f'scores_{NUM_TUPLES}_{MODEL_ALIAS}_{MOVIE_REVIEW_COUNT}.json')
    with open(output_file, 'w') as f:
        json.dump(evaluated_reviews, f, indent=4)

    logger.info(f"Evaluation completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()