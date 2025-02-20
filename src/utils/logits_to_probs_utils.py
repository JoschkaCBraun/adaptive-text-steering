"""
logits_to_probs_utils.py
Utilities for converting logits to probabilities and computing differences.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

def map_answer_tokens_to_ids(tokenizer: AutoTokenizer) -> Dict[str, int]:
    """Get token IDs for all variants of A/B/Yes/No."""
    variants = {
        "A": tokenizer(" (A)").input_ids[3],
        "B": tokenizer(" (B)").input_ids[3],
        "Yes": tokenizer(" Yes").input_ids[2],
        "No": tokenizer(" No").input_ids[2]
    }
    return variants

def extract_token_type(answer: str) -> str:
    """
    Extract the token type (A/B/Yes/No) from an answer string.
    
    Args:
        answer: Answer string containing the token type
        
    Returns:
        str: The extracted token type (A, B, Yes, or No)
        
    Raises:
        ValueError: If no valid token type is found in the answer
    """
    for token_type in ["A", "B", "Yes", "No"]:
        if f" ({token_type})" in answer or f" {token_type}" in answer:
            return token_type
    raise ValueError(f"No valid token type found in answer: {answer}")

def get_answer_token_ids(
    matching_answer: str,
    non_matching_answer: str,
    tokenizer: AutoTokenizer
) -> Tuple[int, int]:
    """
    Get token IDs for both matching and non-matching answers.
    
    Args:
        matching_answer: The matching answer string
        non_matching_answer: The non-matching answer string
        tokenizer: The tokenizer to use
        
    Returns:
        Tuple[int, int]: Token IDs for (matching, non-matching) answers
    """
    token_variants = map_answer_tokens_to_ids(tokenizer)
    matching_type = extract_token_type(matching_answer)
    non_matching_type = extract_token_type(non_matching_answer)
    
    return token_variants[matching_type], token_variants[non_matching_type]

def get_token_probability(
    logits: torch.Tensor,
    answer: str,
    tokenizer: AutoTokenizer
) -> float:
    """
    Compute probability for matching tokens.
    
    Args:
        logits: Logits tensor from model output
        token_variants: Dictionary mapping token types to their IDs
        answer: Answer string containing the token type to match
        
    Returns:
        float: Combined probability for the matching token type
    """
    token_type = extract_token_type(answer)
    token_variants = map_answer_tokens_to_ids(tokenizer)
    token_id = token_variants[token_type]
        
    logits = logits.unsqueeze(0) if len(logits.shape) == 1 else logits
    probs = F.softmax(logits, dim=-1)
    token_prob = probs[:, token_id].item()
    
    return token_prob

def compute_probability_differences(
    matching_logits: torch.Tensor,
    non_matching_logits: torch.Tensor,
    matching_answers: List[str]
) -> List[float]:
    """
    Compute probability differences between matching and non-matching samples.
    
    Args:
        matching_logits: Tensor of logits from matching scenarios
        non_matching_logits: Tensor of logits from non-matching scenarios
        matching_answers: List of answers for matching scenarios
        
    Returns:
        List[float]: List of probability differences between matching and non-matching scenarios
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
    differences = []
    for m_logit, nm_logit, m_answer in zip(matching_logits, non_matching_logits, matching_answers):
        m_prob = get_token_probability(m_logit, m_answer, tokenizer)
        nm_prob = get_token_probability(nm_logit, m_answer, tokenizer)
        differences.append(m_prob - nm_prob)
    
    return differences
