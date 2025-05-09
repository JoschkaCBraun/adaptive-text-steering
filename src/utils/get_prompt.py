"""
get_prompt.py
This file contains functions to generate prompts for article summarization.
"""
import logging
from typing import Optional, Any, List

from src.utils.lda_utils import get_topic_words
from src.utils.validate_inputs import validate_behavior_type
# Configure logger
logging.basicConfig(level=logging.INFO) # Basic config for demonstration
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_SUMMARY_INSTRUCTION = "Write a three sentence summary of the article"
PROMPT_TEMPLATE = '{instruction}\nArticle:\n{article}\nSummary:\n'

def get_newts_summary_prompt(
    article: str,
    behavior_type: str = None,
    use_behavior_encouraging_prompt: bool = False,
    encourage_positive_sentiment: bool = True,
    encourage_toxicity: bool = True,
    encourage_simplicity: bool = True,
    lda: Optional[Any] = None,
    tid: Optional[int] = None,
    num_topic_words: Optional[int] = None
    ) -> str:

    """
    Constructs a summary prompt, optionally focusing on a specific behavior.
    
    Args:
        article: The article text to be summarized.
        behavior_type: The type of behavior to focus on (e.g., "topic", "sentiment", "toxicity", "readability").
        use_behavior_encouraging_prompt: Whether to enhance the prompt with a behavior-encouraging instruction.
        encourage_positive_sentiment: Whether to encourage positive sentiment in the summary.
        encourage_toxicity: Whether to encourage toxic language in the summary (only used with toxicity behavior).
        encourage_simplicity: Whether to encourage simple language in the summary (only used with readability behavior).
        lda: The LDA model used to determine topic focus (required if use_topic=True).
        tid: The topic identifier for the focus topic (required if use_topic=True).
        num_topic_words: The number of topic words to include (required if use_topic=True).
    
    Returns:
        A string containing the structured prompt for summary generation.

    Raises:
        ValueError: If use_behavior_encouraging_prompt is True but required parameters are missing.
        Exception: For errors during prompt formatting.
    """
    if use_behavior_encouraging_prompt:
        if behavior_type is None:
            raise ValueError("behavior_type must be provided if use_behavior_encouraging_prompt is True")
        else:
            validate_behavior_type(behavior_type)

    try:
        instruction = BASE_SUMMARY_INSTRUCTION

        if not use_behavior_encouraging_prompt:
            prompt = _build_prompt(instruction=instruction, article=article)

        elif behavior_type == "topic":
            if lda is None or tid is None or num_topic_words is None:
                raise ValueError(
                    "lda, tid, and num_topic_words must be provided if behavior_type is 'topic' and use_behavior_encouraging_prompt is True"
                )
            # Ensure num_topic_words is positive
            if num_topic_words <= 0:
                 raise ValueError("num_topic_words must be a positive integer")

            top_words = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
            if not top_words:
                 logger.warning(f"No topic words found for tid {tid}. Proceeding without topic focus.")
            else:
                topic_description = ", ".join(top_words)
                instruction += f" focusing on the topic related to: {topic_description}"
            prompt = _build_prompt(instruction=instruction, article=article)    

        elif behavior_type == "sentiment":
            if encourage_positive_sentiment:
                sentiment_focus = " emphasizing the positive outcomes, optimistic viewpoints, or favorable details presented in the article"
            else:
                sentiment_focus = " emphasizing the negative consequences, criticisms, concerns, or problematic details expressed in the article"
            instruction += sentiment_focus
            prompt = _build_prompt(instruction=instruction, article=article)
        
        elif behavior_type == "toxicity":
            if encourage_toxicity:
                instruction += " using toxic and harmful language"
            else:
                instruction += " while avoiding any toxic or harmful language"
            prompt = _build_prompt(instruction=instruction, article=article)
            
        elif behavior_type == "readability":
            if encourage_simplicity:
                instruction += " using very simple and easy to understand language"
            else:
                instruction += " using very complex and sophisticated language"
            prompt = _build_prompt(instruction=instruction, article=article)
        
        else:
            logger.warning(f"Behavior type {behavior_type} is not implemented yet. Generating neutral prompt.")
            prompt = get_newts_summary_prompt(article=article)

        return prompt

    except ValueError as ve: # Catch specific expected errors
        logger.error(f"Configuration error generating topic summary prompt: {ve}")
        raise 
    except Exception as e:
        logger.error(f"Error generating newts summary prompt: {e}")
        raise

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
    instruction = instruction + "."
    return PROMPT_TEMPLATE.format(instruction=instruction, article=article)

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    sample_article = "The weather was surprisingly sunny, though forecasts predicted rain. Local businesses enjoyed increased foot traffic. However, farmers expressed concern about the lack of precipitation for their crops."

    # Example 1: Basic Summary
    prompt1 = get_newts_summary_prompt(article=sample_article, behavior_type="topic")
    print("--- Basic Prompt ---")
    print(prompt1)

    # Example 3: Positive Sentiment Focus
    prompt3 = get_newts_summary_prompt(
        sample_article, 
        behavior_type="sentiment", 
        use_behavior_encouraging_prompt=True,
        encourage_positive_sentiment=True
    )
    print("\n--- Positive Sentiment Prompt ---")
    print(prompt3)

    # Example 4: Negative Sentiment Focus
    prompt4 = get_newts_summary_prompt(
        sample_article, 
        behavior_type="sentiment", 
        use_behavior_encouraging_prompt=True,
        encourage_positive_sentiment=False
    )
    print("\n--- Negative Sentiment Prompt ---")
    print(prompt4)

    # Example 5: Toxicity Focus (avoiding)
    prompt5 = get_newts_summary_prompt(
        sample_article, 
        behavior_type="toxicity", 
        use_behavior_encouraging_prompt=True,
        encourage_toxicity=False
    )
    print("\n--- Toxicity Avoidance Prompt ---")
    print(prompt5)

    # Example 6: Toxicity Focus (encouraging)
    prompt6 = get_newts_summary_prompt(
        sample_article,
        behavior_type="toxicity",
        use_behavior_encouraging_prompt=True,
        encourage_toxicity=True
    )
    print("\n--- Toxicity Encouragement Prompt ---")
    print(prompt6)

    # Example 7: Readability Focus (simple)
    prompt7 = get_newts_summary_prompt(
        sample_article,
        behavior_type="readability",
        use_behavior_encouraging_prompt=True,
        encourage_simplicity=True
    )
    print("\n--- Simple Language Prompt ---")
    print(prompt7)

    # Example 8: Readability Focus (complex)
    prompt8 = get_newts_summary_prompt(
        sample_article,
        behavior_type="readability",
        use_behavior_encouraging_prompt=True,
        encourage_simplicity=False
    )
    print("\n--- Complex Language Prompt ---")
    print(prompt8)