'''
apply_steering_vectors_and_prompting_to_NEWTS.py
This script generates NEWTS summaries for a set of articles from the NEWTS dataset.
During generation, the model is steered towards a specific behavior using a steering vector.
Additionally, for positive steering strengths, the model is encouraged to generate summaries 
in alignment with the steering vector using a behavior-encouraging prompt.
For negative steering strengths, the model is steered away from the behavior using a 
behavior-discouraging prompt. And for steering strength 0, the model is not steered
(steering strength = 0) and a neutral prompt is used.

For topic behavior:
- Positive steering: Encourages topic tid1
- Negative steering: Encourages topic tid2
- Zero steering: Neutral prompt
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
from steering_vectors import SteeringVector

# Local imports
from config.experiment_config import ExperimentConfig
from src.utils.load_models_and_tokenizers import load_model_and_tokenizer
from src.train_vectors.get_steering_vector import get_steering_vector
from src.utils import generate_text_with_steering_vector
from src.utils.get_prompt import get_newts_summary_prompt
from src.utils.load_datasets import load_newts_dataframe, NEWTSDataset
from src.utils.lda_utils import load_lda

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

def get_prompt_for_steering_strength(
    article: str,
    behavior_type: str,
    strength: float,
    tid1: int = None,
    tid2: int = None,
    config: ExperimentConfig = None,
    lda_model = None
) -> str:
    """
    Get the appropriate prompt based on behavior type and steering strength.
    
    Args:
        article: The article text to be summarized
        behavior_type: The type of behavior to focus on
        strength: The steering strength
        tid1: First topic ID (for topic behavior)
        tid2: Second topic ID (for topic behavior)
        config: The experiment configuration
        lda_model: Pre-loaded LDA model
        
    Returns:
        A string containing the appropriate prompt
    """
    if behavior_type == "topic":
        # Use the pre-loaded LDA model
        if lda_model is None:
            logger.warning("Failed to load LDA model. Topic-focused prompts might not work as expected.")
            return get_newts_summary_prompt(
                article=article,
                behavior_type=behavior_type,
                use_behavior_encouraging_prompt=False,
                config=config
            )
            
        num_topic_words = getattr(config, 'NUM_TOPIC_WORDS', 25)  # Default to 25 if not specified
        
        if strength > 0:
            # Encourage tid1 for positive steering
            return get_newts_summary_prompt(
                article=article,
                behavior_type=behavior_type,
                use_behavior_encouraging_prompt=True,
                lda=lda_model,
                tid=tid1,
                num_topic_words=num_topic_words,
                config=config
            )
        elif strength < 0:
            # Encourage tid2 for negative steering
            return get_newts_summary_prompt(
                article=article,
                behavior_type=behavior_type,
                use_behavior_encouraging_prompt=True,
                lda=lda_model,
                tid=tid2,
                num_topic_words=num_topic_words,
                config=config
            )
        else:
            # Neutral prompt for zero steering
            return get_newts_summary_prompt(
                article=article,
                behavior_type=behavior_type,
                use_behavior_encouraging_prompt=False,
                config=config
            )
    elif behavior_type == "sentiment":
        return get_newts_summary_prompt(
            article=article,
            behavior_type=behavior_type,
            use_behavior_encouraging_prompt=strength != 0,
            encourage_positive_sentiment=strength > 0,
            config=config
        )
    elif behavior_type == "toxicity":
        return get_newts_summary_prompt(
            article=article,
            behavior_type=behavior_type,
            use_behavior_encouraging_prompt=strength != 0,
            encourage_toxicity=strength > 0,
            config=config
        )
    elif behavior_type == "readability":
        return get_newts_summary_prompt(
            article=article,
            behavior_type=behavior_type,
            use_behavior_encouraging_prompt=strength != 0,
            encourage_simplicity=strength > 0,
            config=config
        )
    else:
        # Default case - neutral prompt
        return get_newts_summary_prompt(
            article=article,
            behavior_type=behavior_type,
            use_behavior_encouraging_prompt=False,
            config=config
        )

def generate_newts_summaries_with_steering_and_prompting(
    config: ExperimentConfig,
    behavior_type: str,
    model_alias: str,
    load_test_set: bool,
    num_articles: int,
    num_samples: int,
    representation_type: str,
    steering_layers: List[int], 
    language: str = "en",
    pairing_type: str = None) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Generates NEWTS summaries for a set of articles using both steering vectors and prompt engineering.
    
    Args:
        config: Configuration object with experiment settings
        behavior_type: The type of behavior to steer the model towards
        model_alias: The alias of the model to use
        load_test_set: Whether to load the test set
        num_articles: The number of articles to summarize
        num_samples: The number of samples to use for steering vector
        representation_type: Type of representation used
        steering_layers: The layer to apply the steering vector to
        language: The language of the steering vector training data
        pairing_type: The type of pairing to use for steering vector
        
    Returns:
        Dictionary containing experiment metadata and generated summaries
    """
    if behavior_type not in config.VALID_BEHAVIOR_TYPES:
        raise ValueError(f"Invalid behavior type: {behavior_type}")
    
    tokenizer, model, device = load_model_and_tokenizer(model_alias=model_alias)
    max_new_tokens = config.MAX_NEW_TOKENS
    model.eval()
    
    # Load the dataset directly as a DataFrame instead of using a DataLoader with batching
    df = load_newts_dataframe(num_articles=num_articles, load_test_set=False)
    dataset = NEWTSDataset(dataframe=df)

    # Load LDA model once if needed
    lda_model = None
    if behavior_type == "topic":
        lda_model = load_lda()
        if lda_model is None:
            logger.warning("Failed to load LDA model. Topic-focused prompts might not work as expected.")

    if behavior_type != "topic":
        steering_vector: SteeringVector = get_steering_vector(
            behavior_type=behavior_type,
            model_alias=model_alias,
            num_samples=num_samples,
            representation_type=representation_type,
            steering_layers=steering_layers,
            language=language,
            pairing_type=None,
            tid=None
        )
    else:
        steering_vector = None

    experiment_information = {
        "model_alias":         model_alias,
        "behavior_type":       behavior_type,
        "load_test_set":       load_test_set,
        "num_articles":        num_articles,
        "max_new_tokens":      max_new_tokens,
        "representation_type": representation_type,
        "language":            language,
        "steering_layers":     steering_layers,
        "pairing_type":        pairing_type,
        "num_samples":         num_samples,
        "steering_strengths":  config.STEERING_STRENGTHS,
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results: Dict[str, Any] = {"experiment_information": experiment_information}
    generated_summaries: Dict[str, Dict[str, Any]]  = {}
    
    try:
        with torch.no_grad():
            for article_index in range(min(len(dataset), num_articles)):
                article_data = dataset[article_index]
                article_idx = int(article_data['article_idx']) 
                docId = article_data['docId']
                article = article_data['article']
                tid1 = article_data['tid1']
                tid2 = article_data['tid2']
                summary1 = article_data['summary1']
                summary2 = article_data['summary2']
                
                # Get prompts for different steering strengths
                prompts = {
                    str(strength): get_prompt_for_steering_strength(
                        article=article,
                        behavior_type=behavior_type,
                        strength=strength,
                        tid1=tid1,
                        tid2=tid2,
                        config=config,
                        lda_model=lda_model
                    )
                    for strength in config.STEERING_STRENGTHS
                }
                
                summary_entry: Dict[str, Any] = {
                    'docId': docId,
                    'article_idx': article_idx,
                    'article': article,
                    'tid1': tid1,
                    'tid2': tid2,
                    'summary1': summary1,
                    'summary2': summary2,
                    'prompts': prompts,
                    'summaries': {}
                }

                if behavior_type == "topic":
                    steering_vector = get_steering_vector(
                        behavior_type="topic",
                        model_alias=model_alias,
                        representation_type=representation_type,
                        num_samples=num_samples,
                        steering_layers=steering_layers,
                        language=language,
                        pairing_type=pairing_type,
                        tid=tid1
                    )
                
                # Generate summaries with different steering strengths
                for strength in config.STEERING_STRENGTHS:
                    try:
                        summary = generate_text_with_steering_vector(
                            model=model, tokenizer=tokenizer, prompt=prompts[str(strength)],
                            steering_vector=steering_vector, steering_strength=strength,
                            device=device, max_new_tokens=max_new_tokens)
                            
                        summary_entry['summaries'][str(strength)] = summary
                    except Exception as e:
                        logger.error(f"Error generating summary for article {article_idx} with strength {strength}: {str(e)}")
                        summary_entry['summaries'][str(strength)] = f"Error: {str(e)}"
                
                generated_summaries[str(article_idx)] = summary_entry
                logger.info(f"Completed article {len(generated_summaries)}/{num_articles}")
        
        logger.info(f"Generated {len(generated_summaries)} summaries.")
        results['generated_summaries'] = generated_summaries
        
        # Save results to file
        dataset_name = "NEWTS_train" if load_test_set else "NEWTS_test"
        _save_results(results, behavior_type, model_alias, dataset_name, num_articles,
                      representation_type)
        
        return results
    
    except Exception as e:
        logger.error(f"Error generating summaries: {e}")
        raise

def _save_results(
    results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
    behavior_type: str,
    model_alias: str,
    dataset_name: str,
    num_articles: int,
    representation_type: str,
) -> None:
    """
    Save the generated summaries to a JSON file.
    
    Args:
        results: The results dictionary containing experiment info and summaries
        model_alias: The model used
        dataset_name: The dataset used
        num_articles: Number of articles processed
        representation_type: Type of representation used
    """
    # get NEWTS_SUMMARIES_PATH from environment variable
    NEWTS_SUMMARIES_PATH = os.getenv("NEWTS_SUMMARIES_PATH")
    os.makedirs(NEWTS_SUMMARIES_PATH, exist_ok=True)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{behavior_type}_summaries_{model_alias}_{dataset_name}_{num_articles}_articles_{representation_type}_{timestamp}.json"
        filepath = os.path.join(NEWTS_SUMMARIES_PATH, f"{behavior_type}_vectors_with_prompts")
        # make directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main() -> None:
    # test the function
    config = ExperimentConfig()
    behavior_type = "readability"
    model_alias = "llama3_3b"
    load_test_set = False
    num_articles = 100
    representation_type = "words"
    language = "en"
    steering_layers = [8]
    num_samples = config.BEHAVIOR_WORDS_NUM_SAMPLES
    pairing_type = "against_random_topic_representation"  # Set the pairing type

    results = generate_newts_summaries_with_steering_and_prompting(
        config=config,
        behavior_type=behavior_type,
        model_alias=model_alias,
        load_test_set=load_test_set,
        num_articles=num_articles,
        num_samples=num_samples,
        representation_type=representation_type,
        language=language,
        steering_layers=steering_layers,
        pairing_type=pairing_type)

if __name__ == "__main__":
    main()










