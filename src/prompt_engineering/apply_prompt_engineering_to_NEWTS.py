'''
apply_prompt_engineering_to_NEWTS.py
This script generates NEWTS summaries for a set of articles from the NEWTS dataset.
During generation, the model output is influenced by different prompt engineering strategies.
'''
# Standard library imports
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Union, Any

# Third-party imports
import torch
import numpy as np

# Local imports
from config.experiment_config import ExperimentConfig
from src.utils.load_models_and_tokenizers import load_model_and_tokenizer
from src.utils import generate_text_without_steering_vector, load_lda
from src.utils.get_prompt import get_newts_summary_prompt
from src.utils.load_datasets import load_newts_dataframe, NEWTSDataset

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    '''Custom JSON encoder to handle numpy types.'''
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyEncoder, self).default(o)
    


def generate_newts_summaries(
    config: ExperimentConfig,
    model_alias: str,
    load_test_set: bool,
    num_articles: int,
    language: str = "en",
    ) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Generates NEWTS summaries for a set of articles using various prompt engineering techniques.
    
    Args:
        config: Configuration object with experiment settings.
        model_alias: The alias of the model to use.
        load_test_set: Whether to load the test set or the training/validation set.
        num_articles: The number of articles to summarize.
        language: The language (primarily for potential language-specific LDA models, though not directly used in prompts yet).
        
    Returns:
        Dictionary containing experiment metadata and generated summaries with their prompts.
    """
    tokenizer, model, device = load_model_and_tokenizer(model_alias=model_alias)
    max_new_tokens = config.MAX_NEW_TOKENS
    num_topic_words = config.NUM_TOPIC_WORDS
    model.eval()
    
    # Load LDA model
    lda_model = load_lda()
    if lda_model is None:
        logger.warning("Failed to load LDA model. Topic-focused prompts might not work as expected.")

    # Load the dataset
    # The problem description implies load_test_set=False loads train, but the _save_results uses a ternary that suggests opposite.
    # Adhering to the _save_results logic: load_test_set=True -> "NEWTS_test"
    df = load_newts_dataframe(num_articles=num_articles, load_test_set=load_test_set)
    dataset = NEWTSDataset(dataframe=df)

    experiment_information = {
        "experiment_name":     "prompt_engineering",
        "model_alias":         model_alias,
        "load_test_set":       load_test_set,
        "num_articles":        num_articles,
        "max_new_tokens":      max_new_tokens,
        "language":            language,
        "num_topic_words_for_prompt": num_topic_words,
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results: Dict[str, Any] = {"experiment_information": experiment_information}
    generated_summaries_dict: Dict[str, Dict[str, Any]]  = {}
    
    try:
        with torch.no_grad():
            for article_idx_loop in range(min(len(dataset), num_articles)):
                article_data = dataset[article_idx_loop]
                article_actual_idx = int(article_data['article_idx']) 
                docId = article_data['docId']
                article = article_data['article']
                tid1 = article_data['tid1']
                tid2 = article_data['tid2']
                original_summary1 = article_data['summary1']
                original_summary2 = article_data['summary2']
                
                current_article_results: Dict[str, Any] = {
                    'docId': docId,
                    'article_idx': article_actual_idx,
                    'article': article,
                    'tid1': tid1,
                    'tid2': tid2,
                    'original_summary1': original_summary1,
                    'original_summary2': original_summary2,
                    'summaries': {} # This will hold behavior-specific results
                }

                # 1. Generate Neutral Prompt & Summary (once per article)
                neutral_prompt_text = ""
                neutral_summary_text = ""
                try:
                    neutral_prompt_text = get_newts_summary_prompt(article=article) # behavior_type=None, use_behavior_encouraging_prompt=False
                    neutral_summary_text = generate_text_without_steering_vector(
                        model=model, tokenizer=tokenizer, prompt=neutral_prompt_text,
                        device=device, max_new_tokens=max_new_tokens
                    )
                except Exception as e:
                    logger.error(f"Error generating neutral summary for article {article_actual_idx}: {str(e)}")
                    neutral_prompt_text = neutral_prompt_text or "Error in prompt generation"
                    neutral_summary_text = f"Error: {str(e)}"
                
                neutral_result_package = {"prompt": neutral_prompt_text, "summary": neutral_summary_text}

                # 2. Iterate through behavior types
                for behavior_name in config.VALID_BEHAVIOR_TYPES:
                    behavior_specific_outputs: Dict[str, Dict[str, str]] = {}
                    
                    # Common entry for all behaviors
                    behavior_specific_outputs["neutral"] = neutral_result_package

                    try:
                        if behavior_name == "topic":
                            # Tid1
                            prompt_t1 = get_newts_summary_prompt(article=article, behavior_type="topic", use_behavior_encouraging_prompt=True, lda=lda_model, tid=tid1, num_topic_words=num_topic_words)
                            summary_t1 = generate_text_without_steering_vector(model, tokenizer, prompt_t1, device, max_new_tokens)
                            behavior_specific_outputs["topic_tid1_encouraged"] = {"prompt": prompt_t1, "summary": summary_t1}
                            # Tid2
                            prompt_t2 = get_newts_summary_prompt(article=article, behavior_type="topic", use_behavior_encouraging_prompt=True, lda=lda_model, tid=tid2, num_topic_words=num_topic_words)
                            summary_t2 = generate_text_without_steering_vector(model, tokenizer, prompt_t2, device, max_new_tokens)
                            behavior_specific_outputs["topic_tid2_encouraged"] = {"prompt": prompt_t2, "summary": summary_t2}

                        elif behavior_name == "sentiment":
                            # Positive
                            prompt_pos = get_newts_summary_prompt(article=article, behavior_type="sentiment", use_behavior_encouraging_prompt=True, encourage_positive_sentiment=True)
                            summary_pos = generate_text_without_steering_vector(model, tokenizer, prompt_pos, device, max_new_tokens)
                            behavior_specific_outputs["sentiment_positive_encouraged"] = {"prompt": prompt_pos, "summary": summary_pos}
                            # Negative
                            prompt_neg = get_newts_summary_prompt(article=article, behavior_type="sentiment", use_behavior_encouraging_prompt=True, encourage_positive_sentiment=False)
                            summary_neg = generate_text_without_steering_vector(model, tokenizer, prompt_neg, device, max_new_tokens)
                            behavior_specific_outputs["sentiment_negative_encouraged"] = {"prompt": prompt_neg, "summary": summary_neg}
                        
                        elif behavior_name == "toxicity":
                            # Encourage Toxic
                            prompt_enc_tox = get_newts_summary_prompt(article=article, behavior_type="toxicity", use_behavior_encouraging_prompt=True, encourage_toxicity=True)
                            summary_enc_tox = generate_text_without_steering_vector(model, tokenizer, prompt_enc_tox, device, max_new_tokens)
                            behavior_specific_outputs["toxicity_encouraged"] = {"prompt": prompt_enc_tox, "summary": summary_enc_tox}
                            # Avoid Toxic
                            prompt_avoid_tox = get_newts_summary_prompt(article=article, behavior_type="toxicity", use_behavior_encouraging_prompt=True, encourage_toxicity=False)
                            summary_avoid_tox = generate_text_without_steering_vector(model, tokenizer, prompt_avoid_tox, device, max_new_tokens)
                            behavior_specific_outputs["toxicity_avoided"] = {"prompt": prompt_avoid_tox, "summary": summary_avoid_tox}

                        elif behavior_name == "readability":
                            # Simple
                            prompt_simple = get_newts_summary_prompt(article=article, behavior_type="readability", use_behavior_encouraging_prompt=True, encourage_simplicity=True)
                            summary_simple = generate_text_without_steering_vector(model, tokenizer, prompt_simple, device, max_new_tokens)
                            behavior_specific_outputs["readability_simple_encouraged"] = {"prompt": prompt_simple, "summary": summary_simple}
                            # Complex
                            prompt_complex = get_newts_summary_prompt(article=article, behavior_type="readability", use_behavior_encouraging_prompt=True, encourage_simplicity=False)
                            summary_complex = generate_text_without_steering_vector(model, tokenizer, prompt_complex, device, max_new_tokens)
                            behavior_specific_outputs["readability_complex_encouraged"] = {"prompt": prompt_complex, "summary": summary_complex}
                        
                        current_article_results['summaries'][behavior_name] = behavior_specific_outputs

                    except Exception as e:
                        logger.error(f"Error processing behavior {behavior_name} for article {article_actual_idx}: {str(e)}")
                        # Ensure the entry exists even if there was an error
                        if behavior_name not in current_article_results['summaries']:
                             current_article_results['summaries'][behavior_name] = {}
                        current_article_results['summaries'][behavior_name][f"{behavior_name}_error"] = {"prompt": "Error in prompt generation or execution", "summary": f"Error: {str(e)}"}


                generated_summaries_dict[str(article_actual_idx)] = current_article_results
                logger.info(f"Completed article {article_idx_loop + 1}/{num_articles} (Article ID: {article_actual_idx})")
        
        results['generated_summaries'] = generated_summaries_dict
        logger.info(f"Generated summaries for {len(generated_summaries_dict)} articles.")
        
        # Save results to file
        dataset_name = "NEWTS_test" if load_test_set else "NEWTS_train" # if load_test_set=True means use test set
        _save_results(results=results, model_alias=model_alias, dataset_name=dataset_name, num_articles=num_articles)
        
        return results
    
    except Exception as e:
        logger.error(f"Fatal error in generate_newts_summaries: {e}")
        # Save whatever results have been gathered so far if a major error occurs
        if 'generated_summaries' not in results:
            results['generated_summaries'] = generated_summaries_dict
        if generated_summaries_dict: # Only save if there's something to save
             dataset_name = "NEWTS_test" if load_test_set else "NEWTS_train"
             _save_results(results=results, model_alias=model_alias, dataset_name=dataset_name, num_articles=len(generated_summaries_dict), partial_save=True)
        raise

def _save_results(
    results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
    model_alias: str,
    dataset_name: str,
    num_articles: int,
    partial_save: bool = False,
) -> None:
    """
    Save the generated summaries to a JSON file.
    
    Args:
        results: The results dictionary containing experiment info and summaries.
        model_alias: The model used.
        dataset_name: The dataset used (e.g., "NEWTS_train", "NEWTS_test").
        num_articles: Number of articles processed (or intended to be processed).
        partial_save: Flag to indicate if this is a partial save due to an error.
    """
    NEWTS_SUMMARIES_PATH = os.getenv("NEWTS_SUMMARIES_PATH") # Default to ./outputs if not set
    output_dir = os.path.join(NEWTS_SUMMARIES_PATH, "prompt_engineering")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partial_suffix = "_PARTIAL" if partial_save else ""
        filename = f"prompt_engineering_summaries_{model_alias}_{dataset_name}_{num_articles}_articles{partial_suffix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"Results {'partially ' if partial_save else ''}saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        # Do not raise here to avoid masking the original error if this is part of an exception handler
        if not partial_save: # If it's not already a partial save attempt, then raise
            raise

def main() -> None:
    # test the function
    config = ExperimentConfig()
    model_alias = "llama3_1b"
    load_test_set = False
    num_articles = 250
    language = "en"

    results = generate_newts_summaries(
        config=config,
        model_alias=model_alias,
        load_test_set=load_test_set,
        num_articles=num_articles,
        language=language,)

if __name__ == "__main__":
    main()
