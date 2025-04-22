"""
get_prompt.py
This file contains functions to generate prompts for article summarization.
"""

import logging
from typing import Optional, Any
from src.utils.lda_utils import get_topic_words

# Configure logger
logger = logging.getLogger(__name__)


def get_newts_summary_prompt(
    article: str, 
    use_topic: bool = False, 
    lda: Optional[Any] = None, 
    tid: Optional[int] = None,
    num_topic_words: Optional[int] = None
) -> str:
    """
    Constructs a prompt for model, optionally with a focus on specific topic words.
    
    Args:
        article: The article text to be summarized.
        use_topic: Whether to enhance the prompt with topic information.
        lda: The LDA model used to determine topic focus, if any.
        tid: The topic identifier for the focus topic, if any.
        num_topic_words: The number of topic words to include in the prompt, if any.
        
    Returns:
        A string containing the structured prompt for summary generation.
    """
    try:
        initial_instruction = "Write a three sentence summary of the article"
        
        # Apply topic enhancement if requested and possible
        if use_topic:
            if lda is None or tid is None or num_topic_words is None:
                raise ValueError("lda, tid, and num_topic_words must be provided if use_topic is True")
            top_words = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
            topic_description = ", ".join(top_words)
            topical_instruction = f" focusing on {topic_description}."
        else:
            topical_instruction = "."
            
        prompt = f'{initial_instruction}{topical_instruction}\narticle:\n"{article}"\nsummary:\n'
        return prompt
    except Exception as e:
        logger.error(f"Error generating summary prompt: {e}")
        raise