'''
apply_sentiment_vectors_to_NEWTS.py
This script generates NEWTS summaries for a set of articles from the NEWTS dataset.
During generation, the model is steered towards a specific sentiment using a steering vector.
'''

# Standard library imports
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Union, Any

# Third-party imports
import numpy as np
import torch

# Local imports
from config.experiment_config import ExperimentConfig
from src.utils.load_models_and_tokenizers import load_model_and_tokenizer
from src.train_vectors.get_sentiment_vectors import get_sentiment_vector
from src.utils import generate_text_with_steering_vector
from src.utils.get_prompt import get_newts_summary_sentiment_prompt
from src.utils.load_datasets import load_newts_dataframe, NEWTSDataset

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

def generate_newts_summaries(
    config: ExperimentConfig,
    model_alias: str,
    dataset_name: str,
    num_articles: int,
    num_samples: int,
    representation_type: str,
    language: str,
    steering_layers: List[int], 
    use_topic_prompt: bool = False) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Generates topical abstractive summaries for a set of articles.
    
    Args:
        config: Configuration object with experiment settings
        model_alias: The alias of the model to use
        dataset_name: The name of the dataset to use (train or test)
        num_articles: The number of articles to summarize
        layer_to_steer: The layer to apply the steering vector to
        topic_encoding_type: Type of topic encoding to use for steering vector
        
    Returns:
        Dictionary containing experiment metadata and generated summaries
    """
    tokenizer, model, device = load_model_and_tokenizer(model_alias=model_alias)
    max_new_tokens = config.MAX_NEW_TOKENS
    model.eval()
    
    # Load the dataset directly as a DataFrame instead of using a DataLoader with batching
    df = load_newts_dataframe(num_articles=num_articles, load_test_set=False)
    dataset = NEWTSDataset(dataframe=df)
    
    steering_vector = get_sentiment_vector(
        model_alias=model_alias,
        num_samples=num_samples,
        representation_type=representation_type,
        language=language,
        layers=steering_layers
    )
    
    try:
        # Generate experiment metadata first
        experiment_information = {
            'model_alias': model_alias,
            'dataset_name': dataset_name,
            'num_articles': num_articles,
            'max_new_tokens': max_new_tokens,
            'representation_type': representation_type,
            'language': language,
            'steering_layers': steering_layers,
            'use_topic_prompt': use_topic_prompt,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {'experiment_information': experiment_information}
        generated_summaries: List[Dict[str, Any]] = []
        
        with torch.no_grad():  # Operations inside don't track gradients
            # Process one article at a time instead of in batches
            for article_index in range(min(len(dataset), num_articles)):
                article_data = dataset[article_index]
                
                article_idx = int(article_data['article_idx']) 
                article = article_data['article']
                tid1 = article_data['tid1']
                tid2 = article_data['tid2']
                
                prompt = get_newts_summary_sentiment_prompt(article=article)
                summary_entry: Dict[str, Any] = {
                    'article_idx': article_idx,
                    'tid1': tid1,
                    'tid2': tid2,
                    'article': article,
                    'prompt': prompt,
                    'summaries': {}
                }
                
                # Generate summaries with different steering strengths
                for strength in config.STEERING_STRENGTHS:
                    try:
                        print(f"Generating summary for article {article_idx} with strength {strength}")
                        # Apply steering vector with the current strength
                        summary = generate_text_with_steering_vector(model=model, tokenizer=tokenizer, prompt=prompt, steering_vector=steering_vector, steering_strength=strength, device=device, max_new_tokens=max_new_tokens)
                        print(summary)
                            
                        # Store the summary with this steering strength
                        summary_entry['summaries'][str(strength)] = summary
                        logger.info(f"Generated summary for article {article_idx} with strength {strength}")
                    except Exception as e:
                        logger.error(f"Error generating summary for article {article_idx} with strength {strength}: {str(e)}")
                        summary_entry['summaries'][str(strength)] = f"Error: {str(e)}"
                
                generated_summaries.append(summary_entry)
                logger.info(f"Completed article {len(generated_summaries)}/{num_articles}")
        
        logger.info(f"Generated {len(generated_summaries)} summaries.")
        results['generated_summaries'] = generated_summaries
        
        # Save results to file
        save_results(results, model_alias, dataset_name, num_articles, representation_type)
        
        return results
    
    except Exception as e:
        logger.error(f"Error generating summaries: {e}")
        raise


def save_results(
    results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
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
    output_dir = os.path.join("data", "results", "sentiment_vectors")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_summaries_{model_alias}_{dataset_name}_{num_articles}_articles_{representation_type}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {filepath}")


def main() -> None:
    # test the function
    config = ExperimentConfig()
    model_alias = "llama3_1b"
    dataset_name = "NEWTS_train"
    num_articles = 3
    representation_type = "sentiment_sentences"
    language = "en"
    steering_layers = [11]
    num_samples = 500
    use_topic_prompt = False

    results = generate_newts_summaries(config=config, model_alias=model_alias, dataset_name=dataset_name, num_articles=num_articles, num_samples=num_samples, representation_type=representation_type, language=language, steering_layers=steering_layers, use_topic_prompt=use_topic_prompt)
    print(results)

if __name__ == "__main__":
    main()