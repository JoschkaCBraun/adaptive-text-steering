"""
This file contains the configuration for the languagemodels used in the project.
"""
from typing import Dict

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    'llama3_1b': {
        'model_name': "meta-llama/Llama-3.2-1B-Instruct",
        'tokenizer_name': "meta-llama/Llama-3.2-1B-Instruct",
    },
    'llama3_3b': {
        'model_name': "meta-llama/Llama-3.2-3B-Instruct",
        'tokenizer_name': "meta-llama/Llama-3.2-3B-Instruct",
    },
    'llama3_8b': {
        'model_name': "meta-llama/Meta-Llama-3.1-8B-Instruct",
        'tokenizer_name': "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    'llama3_70b': {
        'model_name': "meta-llama/Llama-3.3-70B-Instruct",
        'tokenizer_name': "meta-llama/Llama-3.3-70B-Instruct",
    },
    'openelm_270m': {
        'model_name': "apple/OpenELM-270M-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
    },
    'openelm_450m': {
        'model_name': "apple/OpenELM-450M-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
    },
    'openelm_1b': {
        'model_name': "apple/OpenELM-1_1B-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
    },
    'openelm_3b': {
        'model_name': "apple/OpenELM-3B-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
    },   
    'gemma_2b': {
        'model_name': "google/gemma-2b-it",
        'tokenizer_name': "google/gemma-2b-it",
    },
    'gemma_7b': {
        'model_name': 'google/gemma-7b-it',
        'tokenizer_name': 'google/gemma-7b-it',
    },
    'mistral_7b': {
        'model_name': "mistralai/Mistral-7B-Instruct-v0.1",
        'tokenizer_name': "mistralai/Mistral-7B-Instruct-v0.1",
    },
    'falcon_7b': {
        'model_name': "tiiuae/falcon-7b-instruct",
        'tokenizer_name': "tiiuae/falcon-7b-instruct",
    },
}