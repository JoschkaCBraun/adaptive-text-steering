"""
load_models_and_tokenizers.py

Utility functions for loading models from the transformers AutoModelForCausalLM class
and tokenizers from AutoTokenizer class.
"""
# Standard library imports
import logging
import os
from typing import Optional, Tuple

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local imports
from src.utils import get_device
from config.model_config import MODEL_CONFIGS

# Configure logging at the module level
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
def load_model_and_tokenizer(model_alias: str, device_map: Optional[str] = "auto"
                             ) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """
    Loads the model and tokenizer for a given model alias.

    Args:
        model_alias (str): The alias of the model to be loaded.
        device_map (Optional[str]): The device on which the model is to be loaded.

    Returns:
        Tuple containing the tokenizer, model, and device.
    """
    try:
        model_details = get_model_info(model_alias)
        model_name = model_details.get('model_name')
        tokenizer_name = model_details.get('tokenizer_name')
        device = get_device(device_map=device_map)
        logger.info(f"Loading model: {model_name} and tokenizer: {tokenizer_name}")
        
        hf_auth_token = os.getenv('HF_AUTH_TOKEN')
        torch_dtype = torch.float16 if device.type != "cpu" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_auth_token,
            trust_remote_code=True).to(device)
        
        tokenizer = _load_tokenizer(tokenizer_name=tokenizer_name)
        
        # Ensure the tokenizer has a pad token, set to eos token if not
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))

        logger.info(f"Loaded model: {model_name} and tokenizer: {tokenizer_name}")

    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise
    return tokenizer, model, device

def load_tokenizer(model_alias: str, device_map: Optional[str] = "auto"
                   ) -> Tuple[AutoTokenizer, torch.device]:
    """
    Loads the tokenizer for a given model alias.

    Args:
        model_alias (str): The alias of the model to be loaded.
        device_map (Optional[str]): The device on which the tokenizer is to be loaded.

    Returns:
        Tuple containing the tokenizer and device.
    """
    try:
        model_details = get_model_info(model_alias)
        tokenizer_name = model_details.get('tokenizer_name')
        device = get_device(device_map=device_map)
        logger.info(f"Loading tokenizer: {tokenizer_name}")

        tokenizer = _load_tokenizer(tokenizer_name=tokenizer_name)

        logger.info(f"Loaded tokenizer: {tokenizer_name}")
        return tokenizer, device
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def _load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """
    Helper function to load a tokenizer from the pretrained repository.

    Args:
        tokenizer_name (str): The name of the tokenizer to load.
        hf_auth_token (Optional[str]): Hugging Face authentication token.

    Returns:
        An instance of AutoTokenizer.
    """
    hf_auth_token = os.getenv('HF_AUTH_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
        token=hf_auth_token,
        padding_side='left'
    )
    return tokenizer

def get_model_info(model_alias: str) -> dict:
    """
    Retrieves model and tokenizer names for a given model alias.
    
    Args:
        model_alias (str): The alias of the model.
        
    Returns:
        Dictionary with keys 'model_name' and 'tokenizer_name'.
        
    Raises:
        ValueError: If the model alias is not found in MODEL_CONFIGS.
    """
    if model_alias not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model alias: {model_alias}")
    return MODEL_CONFIGS[model_alias]

# pylint: enable=logging-fstring-interpolation
