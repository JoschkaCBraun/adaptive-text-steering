'''
apply_topic_vectors_to_NEWTS.py
This script generates topical summaries for a set of articles from the NEWTS dataset.
During generation, the model is steered towards a specific topic using a steering vector.

Example usage of steering vector:
with steering_vec.apply(model, multiplier=0.5):
    # the steering vector will be applied at half magnitude
    model.forward(...)
    for prompt, answer in zip(eval_prompts, matching_answers):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        # Use the dictionary format for SteeringVector
        with SteeringVector(layer_activations, "decoder_block").apply(
            model, multiplier=steering_multiplier):
            outputs = model(inputs.input_ids)
            steered_logits = outputs.logits[:, -1, :]
        
        prob = get_token_probability(steered_logits, token_variants, answer)
        results['probs'].append(prob)

'''

# Standard library imports
import os
import sys
import json
import logging
from typing import Optional, List, Dict, Union, Any
from datetime import datetime

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import SteeringVector

# Local imports
from config.experiment_config import ExperimentConfig
from src.utils.load_models_and_tokenizers import load_model_and_tokenizer
from src.train_vectors.get_topic_vectors import get_topic_vector
from src.utils.generation_utils import get_dataloader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_newts_summaries(
    config: ExperimentConfig,
    model_alias: str,
    dataset_name: str,
    num_articles: int,
    topic_vector: torch.Tensor,
    layer_to_steer: int,
    topic_vector_info: Dict[str, Any],
    topic_prompt: str) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]:
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
    # Generate experiment metadata
    experiment_information = {
        'model_alias': model_alias,
        'dataset_name': dataset_name,
        'num_articles': num_articles,
        'layer_to_steer': layer_to_steer,
        'topic_encoding_type': topic_encoding_type,
        'steering_strengths': config.STEERING_STRENGTHS,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Load model and tokenizer
    tokenizer, model, device = load_model_and_tokenizer(model_alias=model_alias)
    model.eval()
    
    # Load dataloader
    dataloader = get_dataloader(dataset_name=dataset_name)
    
    # Set up generation parameters
    generation_config = config.get_generation_config(model_alias=model_alias, tokenizer=tokenizer)
    
    try:
        results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {'experiment_information': experiment_information}
        generated_summaries: List[Dict[str, Any]] = []
        
        max_length = 512  # Maximum input length for the model
        
        with torch.no_grad():  # Operations inside don't track gradients
            for i, batch in enumerate(dataloader):
                if i * config.BATCH_SIZE >= num_articles:
                    break
                
                for index in range(len(batch['article'])):
                    article_idx = batch['article_idx'][index].item()
                    article = batch['article'][index]
                    tid = batch['tid1'][index].item()
                    
                    # Create a summary entry for this article
                    summary_entry: Dict[str, Any] = {
                        'article_idx': article_idx,
                        'tid1': tid,
                        'article': article,
                        'summaries': {}
                    }
                    
                    # Generate the prompt for the article
                    prompt = generate_prompt(article)
                    tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, 
                                               truncation=True, max_length=max_length).to(device)
                    
                    # Get the steering vector for the topic
                    topic_vector = get_topic_vector(
                        model_alias=model_alias,
                        layer=layer_to_steer,
                        topic_representation_type=topic_encoding_type,
                        pairing_type='against_random_topic_representation',
                        tid=tid,
                        num_samples=config.NUM_SAMPLES
                    )
                    
                    # Create a steering vector from the topic vector
                    steering_vector = SteeringVector({layer_to_steer: topic_vector}, "decoder_block")
                    
                    # Generate summaries with different steering strengths
                    for strength in config.STEERING_STRENGTHS:
                        try:
                            # Apply steering vector with the current strength
                            with steering_vector.apply(model, multiplier=strength):
                                outputs = model.generate(
                                    input_ids=tokenized_prompt['input_ids'],
                                    attention_mask=tokenized_prompt['attention_mask'],
                                    **generation_config.to_dict()
                                )
                                
                                # Decode the generated text, skipping prompt tokens
                                decoded_summary = tokenizer.decode(
                                    outputs[0, tokenized_prompt['input_ids'].shape[1]:],
                                    skip_special_tokens=True
                                )
                                
                                # Store the summary with this steering strength
                                summary_entry['summaries'][str(strength)] = decoded_summary
                                logger.info(f"Generated summary for article {article_idx} with strength {strength}")
                        
                        except Exception as e:
                            logger.error(f"Error generating summary for article {article_idx} with strength {strength}: {e}")
                            summary_entry['summaries'][str(strength)] = f"Error: {str(e)}"
                    
                    generated_summaries.append(summary_entry)
                    logger.info(f"Completed article {len(generated_summaries)}/{num_articles}")
        
        logger.info(f"Generated {len(generated_summaries)} summaries.")
        results['generated_summaries'] = generated_summaries
        
        # Save results to file
        save_results(results, config, model_alias, dataset_name, num_articles, 
                    topic_encoding_type, config.STEERING_STRENGTHS)
        
        return results
    
    except Exception as e:
        logger.error(f"Error generating summaries: {e}")
        raise


def save_results(
    results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
    config: ExperimentConfig,
    model_alias: str,
    dataset_name: str,
    num_articles: int,
    topic_encoding_type: str,
    steering_strengths: List[float]
) -> None:
    """
    Save the generated summaries to a JSON file.
    
    Args:
        results: The results dictionary containing experiment info and summaries
        config: The experiment configuration
        model_alias: The model used
        dataset_name: The dataset used
        num_articles: Number of articles processed
        topic_encoding_type: Type of topic encoding used
        steering_strengths: The steering strengths used
    """
    output_dir = os.path.join("data", "results", "topic_vectors")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"topic_summaries_{model_alias}_{dataset_name}_{num_articles}_articles_{topic_encoding_type}_"\
               f"{min(steering_strengths)}_{max(steering_strengths)}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {filepath}")

def main() -> None:
    """
    Main function to run the script.
    """
    config = ExperimentConfig()
    model_alias = config.MODEL_ALIAS
    dataset_name = config.TEST_DATASET_NAME
    num_articles = config.NUM_ARTICLES
    layer_to_steer = 12  # Default layer to steer
    topic_encoding_type = 'topic_words'  # Default topic encoding type
    
    # Generate topical summaries
    generate_newts_summaries(
        config=config,
        model_alias=model_alias,
        dataset_name=dataset_name,
        num_articles=num_articles,
        layer_to_steer=layer_to_steer,
        topic_encoding_type=topic_encoding_type
    )

if __name__ == "__main__":
    main()