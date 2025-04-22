'''
generation_utils.py 
This script provides utility functions for generating summaries using any model from the 
AutoModelForCausalLM class in the transformers library and tokenizer from the AutoTokenizer class.
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

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from config.experiment_config import Config
from utils.load_topic_lda import get_topic_vector, get_topic_words, load_model_and_tokenizer,\
    get_dataloader, load_lda, find_data_dir, get_topic_tokens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    model_alias = Config.MODEL_ALIAS

    generation_config = Config.get_generation_config(model_alias=model_alias, tokenizer=tokenizer)
    if model_alias in ['openelm_270m', 'openelm_450m', 'openelm_1b', 'openelm_3b']:
        max_length = 2048
    else:
        max_length = model.config.max_position_embeddings

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

    return article_summaries


def store_results(summaries: List[Dict[str, str]]) -> None:
    """
    Stores the generated summaries to a file in the results directory.

    :param summaries: The generated summaries to store.
    """

    file_name = (f"{Config.EXPERIMENT_NAME}_{Config.MODEL_ALIAS}_{Config.NUM_ARTICLES}_"
                 f"{Config.MIN_NEW_TOKENS}_{Config.MAX_NEW_TOKENS}_{Config.NUM_BEAMS}.json")
    start_path = os.getcwd()
    data_path = find_data_dir(start_path)
    if not data_path:
        logging.error("Data directory not found. Summaries not stored.")
        return

    results_dir = os.path.join(data_path, 'results_{experiment_name}', file_name)
    with open(results_dir, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=4)
    logging.info("Summaries stored.")
    return
