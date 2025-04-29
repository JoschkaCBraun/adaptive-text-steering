'''
apply_topic_vectors_to_NEWTS.py
This script generates topical summaries for a set of articles from the NEWTS dataset.
During generation, the model is steered towards a specific topic using a steering vector.
'''

# Standard library imports
import os
import json
import logging
from typing import List, Dict, Any, Union
from datetime import datetime

# Third-party imports
import torch
import numpy as np
from steering_vectors import SteeringVector

# Local imports
from config.experiment_config import ExperimentConfig
from src.utils.load_models_and_tokenizers import load_model_and_tokenizer
from src.train_vectors.get_topic_vectors import get_topic_vector
from src.utils.load_datasets import load_newts_dataframe, NEWTSDataset
from src.utils.get_prompt import get_newts_summary_topic_prompt
from src.train_vectors.get_topic_vectors import get_topic_vector
from src.utils import generate_text_with_steering_vector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_newts_topic_summaries(
        config: ExperimentConfig,
        model_alias: str,
        dataset_name: str,
        num_articles: int,
        steering_layers: List[int],
        representation_type: str,
        pairing_type: str,
        num_samples: int,
        language: str = "en",
        use_topic_prompt: bool = False) -> Dict[str, Any]:

    """
    Generates topical abstractive summaries for a set of articles.
    
    Args:
        config: Configuration object with experiment settings
        model_alias: The alias of the model to use
        dataset_name: The name of the dataset to use (train or test)
        num_articles: The number of articles to summarize
        steering_layers: The layer to apply the steering vector to
        representation_type: Type of topic encoding to use for steering vector
        pairing_type: Type of pairing to use for steering vector
        num_samples: The number of samples to use for steering vector
        language: The language of the model
        use_topic_prompt: Whether to use the topic prompt
        
    Returns:
        Dictionary containing experiment metadata and generated summaries
    """
    tokenizer, model, device = load_model_and_tokenizer(model_alias=model_alias)
    max_new_tokens = config.MAX_NEW_TOKENS
    model.eval()

    df = load_newts_dataframe(num_articles=num_articles, load_test_set=("test" in dataset_name))
    dataset = NEWTSDataset(dataframe=df)

    experiment_information = {
        "model_alias":        model_alias,
        "dataset_name":       dataset_name,
        "num_articles":       num_articles,
        "max_new_tokens":     max_new_tokens,
        "representation_type": representation_type,
        "language":           language,
        "steering_layers":    steering_layers,
        "pairing_type":       pairing_type,
        "num_samples":        num_samples,
        "steering_strengths": config.STEERING_STRENGTHS,
        "use_topic_prompt":   use_topic_prompt,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results: Dict[str, Any]               = {"experiment_information": experiment_information}
    generated: Dict[str, Dict[str, Any]]  = {}

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
                
            prompt = get_newts_summary_topic_prompt(article=article, use_topic=use_topic_prompt)
            summary_entry: Dict[str, Any] = {
                'docId': docId,
                'article_idx': article_idx,
                'article': article,
                'tid1': tid1,
                'tid2': tid2,
                'summary1': summary1,
                'summary2': summary2,
                'prompt': prompt,
                'summaries': {}
            }

            prompt  = get_newts_summary_topic_prompt(article=article, use_topic=use_topic_prompt)

            # obtain a dict {layer: vec} for *this* tid
            layer = steering_layers[0]
            steering_vector = get_topic_vector(
                model_alias=model_alias,
                layer=layer,
                topic_representation_type=representation_type,
                pairing_type=pairing_type,
                tid=tid1,
                num_samples=num_samples,
                language=language
            )

            for strength in config.STEERING_STRENGTHS:
                try:
                    summary = generate_text_with_steering_vector(
                        model=model, tokenizer=tokenizer, prompt=prompt,
                        steering_vector=steering_vector, steering_strength=strength, device=device,
                        max_new_tokens=max_new_tokens)
                        
                    summary_entry['summaries'][str(strength)] = summary
                except Exception as e:
                    logger.error(f"Error generating summary for article {article_idx} with strength {strength}: {str(e)}")
                    summary_entry['summaries'][str(strength)] = f"Error: {str(e)}"

            generated[str(article_idx)] = summary_entry
            logging.info(f"Completed article {len(generated)}/{num_articles}")

    results["generated_summaries"] = generated
    _save_topic_results(results, model_alias, dataset_name, num_articles, representation_type)
    return results

def _save_topic_results(results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
                        model_alias: str,
                        dataset_name: str,
                        num_articles: int,
                        representation_type: str) -> None:
    out_dir = os.path.join("data", "results", "topic_vectors")
    os.makedirs(out_dir, exist_ok=True)

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"topic_summaries_{model_alias}_{dataset_name}_{num_articles}_articles_{representation_type}_{timestamp}.json"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main() -> None:
    config = ExperimentConfig()
    model_alias = "llama3_1b"
    dataset_name = "NEWTS_train"
    num_articles = 250
    steering_layers = [8]
    representation_type = "topic_words"
    pairing_type = "against_random_topic_representation"
    num_samples = 5000
    language = "en"

    generate_newts_topic_summaries(
        config=config,
        model_alias=model_alias,
        dataset_name=dataset_name,
        num_articles=num_articles,
        steering_layers=steering_layers,
        representation_type=representation_type,
        pairing_type=pairing_type,
        num_samples=num_samples,
        language=language,
    )

if __name__ == "__main__":
    main()
