'''
generate_topical_newts_summaries.py
This script generates topical summaries for a set of articles from the NEWTS dataset.
During generation, the model is steered towards a specific topic using a steering vector.

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
from typing import Optional, List, Dict, Union

# Third-party imports
import torch
from gensim.models.ldamodel import LdaModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.experiment_config import Config
from src.utils import get_topic_vector, load_model_and_tokenizer,\
    get_dataloader, get_topic_tokens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_newts_summaries(
    config: Config,
    model_alias: str,
    dataset_name: str,
    num_articles: int,
    layer_to_steer: int)-> List[Dict[str, Union[int, str]]]:
    """
    Generates topical abstractive summaries for a set of articles.
    arguments: 
    - model_alias: the alias of the model to use
    - dataset_name: the name of the dataset to use (train or test)
    - num_articles: the number of articles to summarize
    - steering vector
    """

    # initialize experiment hyperparameters
    min_new_tokens = Config.MIN_NEW_TOKENS
    max_new_tokens = Config.MAX_NEW_TOKENS
    num_beams = Config.NUM_BEAMS
    batch_size = Config.BATCH_SIZE

    # load model and tokenizer
    tokenizer, model, device = load_model_and_tokenizer()
    model.eval()

    # load dataloader
    dataloader = get_dataloader(dataset_name=dataset_name)

    try:
        tokenizer, model, device = load_model_and_tokenizer()
        model.eval()

        results = {'experiment_information': experiment_information}
        generated_summaries = []

        with torch.no_grad():  # Operations inside don't track gradients
            for i, batch in enumerate(dataloader):
                if i * batch_size >= num_articles:
                    break

                for index in range(len(batch['article'])):
                    generated_summaries.append(
                        generate_individual_summary(model=model, steering_vector=steering_vector, , batch=batch, index=index, tokenizer=tokenizer,
                                                    device=device, model=model))

        logger.info(f"Generated {len(generated_summaries)} summaries.")
        results['generated_summaries'] = generated_summaries
        store_results(results)
        return results
    except Exception as e:
        logger.error(f"Error generating {experiment_name} summaries: {e}")
        raise

def generate_metadata(config: Config, model_alias: str, dataset_name: str, num_articles: int, steering_vector: torch.Tensor)-> Dict[str, Union[int, str]]:
    '''
    Generates metadata for a set of summaries.
    '''
    metadata = {
        'model_alias': model_alias,
        'dataset_name': dataset_name,
        'num_articles': num_articles,
        'steering_vector': steering_vector
    }
    return metadata

def generate_individual_summary(batch: Dict, index: int, tokenizer: AutoTokenizer, device,
                                model: AutoModelForCausalLM, lda: Optional[LdaModel] = None,
                                ) -> Dict[str, Union[int, str]]:
    '''
    Generate summaries for a single article.
    '''

    experiment_name = Config.EXPERIMENT_CONFIG
    article, article_idx = batch['article'][index], batch['article_idx'][index].item()
    tid1 = batch['tid1'][index].item()
    article_summaries = {'artciel_idx': article_idx,
                         'tid1': tid1}


    if experiment_name in ['baseline', 'topic_vectors']:
        prompt = generate_prompt(article)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                     max_length=max_length).to(device)
        if experiment_name == 'baseline':
            outputs = model.generate(input_ids=tokenized_prompt['input_ids'],
                                    attention_mask=tokenized_prompt['attention_mask'],
                                    **generation_config.to_dict())
        elif experiment_name == 'topic_vectors':

            for topic_encoding_type in Config.TOPIC_VECTORS_CONFIG:
                steering_vector = get_topic_vector(tid=tid1, topic_encoding_type=topic_encoding_type)
                with steering_vector.apply(model):
                    outputs = model.generate(input_ids=tokenized_prompt['input_ids'],
                                            attention_mask=tokenized_prompt['attention_mask'],
                                            **generation_config.to_dict())
                decoded_summary = tokenizer.decode(outputs[:, tokenized_prompt['input_ids'].shape[1]:].squeeze(),
                                           skip_special_tokens=False)
                article_summaries[topic_encoding_type] = decoded_summary

    elif experiment_name == 'prompt_engineering':
        tid1, tid2 = batch['tid1'][index].item(), batch['tid2'][index].item()
        article_summaries['tid1'] = tid1
        article_summaries['tid2'] = tid2
        focus_types = Config.PROMPT_ENGINEERING_CONFIG['focus_types']

        for focus_type in focus_types:
            tid = tid1 if focus_type == 'tid1_focus' else (tid2 if focus_type == 'tid2_focus' else None)
            prompt = generate_prompt(article, lda=lda if tid is not None else None, tid=tid)
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                        max_length=max_length).to(device)
            outputs = model.generate(input_ids=tokenized_prompt['input_ids'],
                                    attention_mask=tokenized_prompt['attention_mask'],
                                    **generation_config.to_dict()
                                    )
            decoded_summary = tokenizer.decode(outputs[:, tokenized_prompt['input_ids'].shape[1]:].squeeze(),
                                            skip_special_tokens=False)
            article_summaries[focus_type] = decoded_summary

    elif experiment_name in ['factor_scaling', 'constant_shift', 'threshold_selection']:
        topic_tokens = get_topic_tokens(tokenizer=tokenizer, lda=lda, tid=tid)

        prompt = generate_prompt(article=article)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                     max_length=max_length).to(device)

        # Set parameters for generation as defined in the config.py file
        generation_config = Config.get_generation_config(model_alias=model_alias,
                                                         tokenizer=tokenizer)
        topical_encouragement = None
        if experiment_name == 'factor_scaling':
            factors = Config.FACTOR_SCALING_CONFIG['scaling_factors']
        elif experiment_name == 'constant_shift':
            factors = Config.CONSTANT_SHIFT_CONFIG['shift_constants']
        elif experiment_name == 'threshold_selection':
            factors = Config.THRESHOLD_SELECTION_CONFIG['selection_thresholds']
            topical_encouragement = Config.THRESHOLD_SELECTION_CONFIG['topical_encouragement']



            # summary = tokenizer.decode(output[tokenized_prompt['input_ids'].shape[1]:], skip_special_tokens=False)
            decoded_summary = tokenizer.decode(output.squeeze(), skip_special_tokens=False)
            article_summaries[str(factor)] = decoded_summary

    return article_summaries

def generate_prompt(article: str) -> str:
    """
    Constructs a prompt for model.
    
    :param article: The article text to be summarized.
    :return: A string containing the structured prompt for summary generation.
    """

    try:
        initial_instruction = "Write a three sentence summary of the article.\narticle:\n"
        prompt = f'{initial_instruction}"{article}"\nsummary:\n'
        return prompt
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise
