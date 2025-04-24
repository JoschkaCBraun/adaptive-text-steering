'''
generation_utils.py 
This script provides utility functions for generating summaries using any model from the 
AutoModelForCausalLM class in the transformers library and tokenizer from the AutoTokenizer class.
'''

# Standard library imports
import os
import sys
import logging
from typing import Dict, Union

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from steering_vectors import SteeringVector

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_text_with_steering_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    steering_vector: SteeringVector,
    steering_strength: float,
    device: torch.device,
    max_new_tokens: int = 256,
    **generation_kwargs
) -> str:
    """
    Generates text using a model with a steering vector applied, returning
    only the newly generated text.

    Args:
        model: The causal language model.
        tokenizer: The tokenizer associated with the model.
        prompt: The input text prompt.
        steering_vector: The SteeringVector object to apply.
        steering_strength: The multiplier for the steering vector's effect.
        device: The torch device (e.g., 'cuda', 'cpu').
        max_new_tokens: The maximum number of new tokens to generate.
        **generation_kwargs: Additional keyword arguments passed to model.generate().

    Returns:
        The newly generated text string, excluding the input prompt.
    """

    # Tokenize input and get length
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs.input_ids
    input_length = input_ids.shape[1]

    # Prepare generation configuration, merging user kwargs with defaults
    # Ensure essential pad/eos tokens are set if not provided by user/tokenizer defaults
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        **generation_kwargs # User-provided kwargs override defaults
    }

    # Apply steering vector context and generate token IDs
    with steering_vector.apply(model, multiplier=steering_strength):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"), # Use attention mask from tokenizer output
            **generation_config
        )

    # Slice output tensor to get only generated token IDs
    # outputs[0] has the full sequence [prompt_tokens, generated_tokens]
    output_ids = outputs[0]
    generated_ids = output_ids[input_length:]

    # Decode only the generated tokens and clean up whitespace
    generated_text = tokenizer.decode(generated_ids.cpu(), skip_special_tokens=True).strip()

    return generated_text