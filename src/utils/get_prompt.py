"""
get_prompt.py
This file contains functions to generate prompts for article summarization.
"""
import logging
from typing import Optional, Any, List

from src.utils.lda_utils import get_topic_words

# Configure logger
logging.basicConfig(level=logging.INFO) # Basic config for demonstration
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_SUMMARY_INSTRUCTION = "Write a three sentence summary of the article"
PROMPT_TEMPLATE = '{instruction}\nArticle:\n{article}\nSummary:\n'

# --- Helper Function ---
def _build_prompt(instruction: str, article: str) -> str:
    """
    Constructs the final prompt string using a standard template.
    
    Args:
        instruction: The specific instruction for the summary task.
        article: The article text.
        
    Returns:
        The formatted prompt string.
    """
    return PROMPT_TEMPLATE.format(instruction=instruction, article=article)

# --- Prompt Generation Functions ---
def get_newts_summary_topic_prompt(
    article: str,
    use_topic: bool = False,
    lda: Optional[Any] = None,
    tid: Optional[int] = None,
    num_topic_words: Optional[int] = None
) -> str:
    """
    Constructs a summary prompt, optionally focusing on specific topic words.
    
    Args:
        article: The article text to be summarized.
        use_topic: Whether to enhance the prompt with topic information.
        lda: The LDA model used to determine topic focus (required if use_topic=True).
        tid: The topic identifier for the focus topic (required if use_topic=True).
        num_topic_words: The number of topic words to include (required if use_topic=True).
        
    Returns:
        A string containing the structured prompt for summary generation.
        
    Raises:
        ValueError: If use_topic is True but lda, tid, or num_topic_words are missing.
        Exception: For errors during topic word retrieval or prompt formatting.
    """
    try:
        instruction = BASE_SUMMARY_INSTRUCTION
        
        # Apply topic enhancement if requested and possible
        if use_topic:
            if lda is None or tid is None or num_topic_words is None:
                raise ValueError(
                    "lda, tid, and num_topic_words must be provided if use_topic is True"
                )
            # Ensure num_topic_words is positive
            if num_topic_words <= 0:
                 raise ValueError("num_topic_words must be a positive integer")

            top_words = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
            if not top_words:
                 logger.warning(f"No topic words found for tid {tid}. Proceeding without topic focus.")
                 instruction += "."
            else:
                topic_description = ", ".join(top_words)
                instruction += f" focusing on the topic related to: {topic_description}."
        else:
            instruction += "." # End the sentence if no topic focus added
            
        return _build_prompt(instruction=instruction, article=article)
        
    except ValueError as ve: # Catch specific expected errors
        logger.error(f"Configuration error generating topic summary prompt: {ve}")
        raise 
    except Exception as e: # Catch other potential errors (e.g., from get_topic_words)
        logger.error(f"Error generating topic summary prompt: {e}")
        raise

def get_newts_summary_sentiment_prompt(
    article: str,
    use_sentiment: bool = False,
    focus_on_positive: bool = True
) -> str:
    """
    Constructs a summary prompt, optionally focusing on sentiment.
    
    Args:
        article: The article text to be summarized.
        use_sentiment: Whether to enhance the prompt with sentiment focus.
        focus_on_positive: If use_sentiment is True, determines whether to focus 
                           on positive (True) or negative (False) aspects.
                           
    Returns:
        A string containing the structured prompt for summary generation.
        
    Raises:
        Exception: For errors during prompt formatting.
    """
    try:
        instruction = BASE_SUMMARY_INSTRUCTION
        
        if use_sentiment:
            if focus_on_positive:
                # Reworded instruction
                sentiment_focus = " emphasizing the positive aspects mentioned."
            else:
                # Reworded instruction
                sentiment_focus = " emphasizing the negative aspects mentioned."
            instruction += sentiment_focus
        else:
            instruction += "." # End the sentence if no sentiment focus added

        # Correctly use the constructed instruction
        return _build_prompt(instruction=instruction, article=article)
        
    except Exception as e:
        logger.error(f"Error generating sentiment summary prompt: {e}")
        raise

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    sample_article = "The weather was surprisingly sunny, though forecasts predicted rain. Local businesses enjoyed increased foot traffic. However, farmers expressed concern about the lack of precipitation for their crops."

    # Example 1: Basic Summary
    prompt1 = get_newts_summary_topic_prompt(sample_article)
    print("--- Basic Prompt ---")
    print(prompt1)

    # Example 2: Topic-Focused Summary (using placeholder LDA)
    # In a real scenario, 'lda_model' would be your trained LDA model instance
    lda_model_stub = "mock_lda_model" 
    topic_id = 1
    num_words = 5
    try:
        prompt2 = get_newts_summary_topic_prompt(
            sample_article, 
            use_topic=True, 
            lda=lda_model_stub, 
            tid=topic_id, 
            num_topic_words=num_words
        )
        print("\n--- Topic-Focused Prompt ---")
        print(prompt2)
    except Exception as e:
        print(f"\nError generating topic prompt: {e}")


    # Example 3: Positive Sentiment Focus
    prompt3 = get_newts_summary_sentiment_prompt(
        sample_article, 
        use_sentiment=True, 
        focus_on_positive=True
    )
    print("\n--- Positive Sentiment Prompt ---")
    print(prompt3)

    # Example 4: Negative Sentiment Focus
    prompt4 = get_newts_summary_sentiment_prompt(
        sample_article, 
        use_sentiment=True, 
        focus_on_positive=False
    )
    print("\n--- Negative Sentiment Prompt ---")
    print(prompt4)