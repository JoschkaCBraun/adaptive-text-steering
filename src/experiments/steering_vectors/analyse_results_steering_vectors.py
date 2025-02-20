'''
analyze_sentiment_vectors.py

This script analyzes the generated movie reviews by evaluating their sentiment,
detecting language, and calculating perplexity as a measure of coherence.
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
from src.utils.load_and_get_utils import load_model_and_tokenizer, get_data_dir
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

def calculate_perplexity(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> float:
    '''Calculate the perplexity of the given text using the provided model and tokenizer.'''
    encodings = tokenizer(text, return_tensors='pt').to(model.device)
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def evaluate_review(review: Dict[str, Any], sentiment_pipeline, lang_model, coherence_model: AutoModelForCausalLM, coherence_tokenizer: AutoTokenizer) -> Dict[str, Any]:
    '''Evaluate a single review by analyzing sentiment, detecting language, and calculating perplexity.'''
    text = review['review']
    
    sentiment_score, sentiment_confidence = analyze_sentiment(text, sentiment_pipeline)
    detected_language = detect_language(text, lang_model)
    perplexity = calculate_perplexity(text, coherence_model, coherence_tokenizer)
    
    review['evaluation'] = {
        'sentiment_score': sentiment_score,
        'sentiment_confidence': sentiment_confidence,
        'detected_language': detected_language,
        'perplexity': perplexity,
        'language_match': detected_language == review['prompt_language']
    }
    
    return review

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