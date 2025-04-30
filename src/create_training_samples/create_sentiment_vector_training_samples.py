"""
create_sentiment_vector_training_samples.py

This file contains the code to create training samples for the sentiment vector.

Text is loaded from datasets/sentiment/sentiment_synthetic_data/ 
Options are sentiment_sentences and sentiment_words.
Return a list of tuples of positive and negative samples.
"""

import os
from typing import List, Tuple, Optional
import pandas as pd

# --- Constants ---
SENTIMENT_SENTENCES = 'sentences'
SENTIMENT_WORDS = 'words'
VALID_REPRESENTATION_TYPES = {SENTIMENT_SENTENCES, SENTIMENT_WORDS}

# Default/Maximum expected samples in the standard files
MAX_SENTENCES = 500
MAX_WORDS = 2800
# --- End Constants ---
def create_sentiment_vector_training_samples(
    representation_type: str,
    num_samples: Optional[int] = None,
    language: str = 'en'
) -> List[Tuple[str, str]]:
    '''
    Load sentiment vector training samples (positive/negative pairs) from a CSV file.

    The function expects the CSV file to have columns named 'positive' and 'negative'.
    It reads data from a path specified by the 'SENTIMENT_DATA_PATH' environment variable.

    Args:
        representation_type (str): Type of sentiment representation
                                   (use constants SENTIMENT_SENTENCES or SENTIMENT_WORDS).
        num_samples (Optional[int]): The maximum number of samples (pairs) to load.
                                     If None, loads the default maximum for the type
                                     (MAX_SENTENCES or MAX_WORDS). Defaults to None.
        language (str): Language code. Currently, only 'en' is fully supported,
                        especially for SENTIMENT_WORDS. Defaults to 'en'.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains
                               a (positive_sample, negative_sample) string pair.

    Raises:
        ValueError: If SENTIMENT_DATA_PATH environment variable is not set.
        ValueError: If representation_type is invalid.
        ValueError: If num_samples is less than or equal to 0.
        ValueError: If requested num_samples exceeds the maximum for the type.
        ValueError: If a non-'en' language is requested for SENTIMENT_WORDS.
        FileNotFoundError: If the expected data file does not exist.
        ValueError: If the data file is found but is empty or lacks required columns.
    '''
    # --- Input Validation ---
    if representation_type not in VALID_REPRESENTATION_TYPES:
        raise ValueError(f"Invalid representation type: '{representation_type}'. "
                         f"Expected one of {VALID_REPRESENTATION_TYPES}")

    if representation_type == SENTIMENT_SENTENCES:
        max_allowed = MAX_SENTENCES
        default_num = MAX_SENTENCES
    else: # SENTIMENT_WORDS
        max_allowed = MAX_WORDS
        default_num = MAX_WORDS
        if language != 'en':
            raise ValueError(f"Language '{language}' is not supported for {SENTIMENT_WORDS}. "
                             f"Only 'en' is available.")

    if num_samples is None:
        num_samples = default_num
    elif not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"Number of samples must be a positive integer, got {num_samples}")
    elif num_samples > max_allowed:
         raise ValueError(f"Requested number of samples ({num_samples}) exceeds the maximum "
                          f"allowed ({max_allowed}) for '{representation_type}'.")

    # --- Environment and Path Setup ---
    data_dir = os.getenv('SENTIMENT_DATA_PATH')
    if data_dir is None:
        raise ValueError("Environment variable 'SENTIMENT_DATA_PATH' is not set.")

    try:
        file_name = get_sentiment_representation_file_name(representation_type, language)
        sub_dir = f"sentiment_{representation_type}"
        file_path = os.path.join(data_dir, sub_dir, file_name)

        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Data file not found at expected path: {file_path}")

    except ValueError as e: # Catch error from get_sentiment_representation_file_name
        raise e # Re-raise it

    # --- Data Loading and Processing ---
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"No data found in file: {file_path}")

        if 'positive' not in df.columns or 'negative' not in df.columns:
            raise ValueError(f"CSV file '{file_path}' must contain 'positive' and 'negative' columns.")

        # Limit the number of samples if requested, *after* loading
        # Use head() to get the top N rows efficiently
        df_limited = df.head(num_samples)

        # Convert columns to list of tuples efficiently
        training_samples = list(zip(df_limited['positive'].astype(str),
                                    df_limited['negative'].astype(str)))

        if not training_samples:
             # This case might occur if num_samples was valid but the file had fewer rows than expected
             print(f"Warning: Loaded 0 samples from {file_path} after requesting {num_samples}. "
                   f"File might contain fewer rows.")
             return []


        return training_samples

    except FileNotFoundError as e:
        raise e # Propagate file not found error
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {file_path}") from None
    except Exception as e:
        # Catch other potential pandas or file reading errors
        raise IOError(f"Error reading or processing file {file_path}: {e}") from e

def get_sentiment_representation_file_name(representation_type: str, language: str = 'en') -> str:
    '''
    Construct the sentiment representation file name based on type and language.

    Args:
        representation_type (str): Type of sentiment representation
                                   (use constants SENTIMENT_SENTENCES or SENTIMENT_WORDS).
        language (str): Language code (default: 'en').

    Returns:
        str: File name for the specified representation type and language.

    Raises:
        ValueError: If representation type is invalid.
    '''
    if representation_type == SENTIMENT_SENTENCES:
        # Assumes filename format includes the max number of samples
        return f"sentiment_{representation_type}_{MAX_SENTENCES}_{language}.csv"
    elif representation_type == SENTIMENT_WORDS:
        # Assumes filename format includes the max number of samples
        return f"sentiment_{representation_type}_{MAX_WORDS}_{language}.csv"
    else:
        raise ValueError(f"Invalid representation type: '{representation_type}'. "
                         f"Expected one of {VALID_REPRESENTATION_TYPES}")

def main() -> None:
    """
    Main function for demonstration purposes. Loads examples of sentence and word pairs.
    """
    print("Demonstrating sentiment data loading.")
    print("-" * 30)

    try:
        print(f"Loading {SENTIMENT_SENTENCES} (up to {MAX_SENTENCES})...")
        samples_sent = create_sentiment_vector_training_samples(
            representation_type=SENTIMENT_SENTENCES,
            num_samples=MAX_SENTENCES # Request explicit max
        )
        print(f"Loaded {len(samples_sent)} sentiment sentence pairs.")
        if samples_sent:
            print(f"Example pair: {samples_sent[0]}")

        print("-" * 30)
        print(f"Loading {SENTIMENT_SENTENCES} (requesting 10 samples)...")
        samples_sent_small = create_sentiment_vector_training_samples(
            representation_type=SENTIMENT_SENTENCES,
            num_samples=10
        )
        print(f"Loaded {len(samples_sent_small)} sentiment sentence pairs.")
        if samples_sent_small:
            print(f"Example pair: {samples_sent_small[0]}")


        print("-" * 30)
        print(f"Loading {SENTIMENT_WORDS} (up to {MAX_WORDS}, language='en')...")
        samples_word = create_sentiment_vector_training_samples(
            representation_type=SENTIMENT_WORDS,
            num_samples=MAX_WORDS, # Request explicit max
            language='en'
        )
        print(f"Loaded {len(samples_word)} sentiment word pairs.")
        if samples_word:
            print(f"Example pair: {samples_word[0]}")

    except (ValueError, FileNotFoundError, IOError) as e:
        print(f"\nError during demonstration: {e}")
        print("Please ensure:")
        print("1. The 'SENTIMENT_DATA_PATH' environment variable is set correctly.")
        print(f"2. The expected subdirectories ('{SENTIMENT_SENTENCES}', '{SENTIMENT_WORDS}') exist.")
        print("3. The required CSV files (e.g., 'sentiment_sentences_500_en.csv') are present and readable.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
