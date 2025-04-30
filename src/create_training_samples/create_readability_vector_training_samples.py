from typing import List, Tuple
import json
import csv
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def create_readability_vector_training_samples(
    num_samples: int,
    representation_type: str,
    language: str = 'en'
) -> List[Tuple[str, str]]:
    """
    Create training samples for the readability vector.

    representation_type: 'sentences' or 'words'
    language: only 'en' is supported
    Returns a list of tuples: (simple, complex)
    """
    if language != 'en':
        logging.error("Unsupported language requested: %s", language)
        raise ValueError(f"Data not available for language: {language}")

    if representation_type == 'sentences':
        return load_readability_sentences(num_samples)
    elif representation_type == 'words':
        return load_readability_words(num_samples)
    else:
        logging.error("Invalid representation type: %s", representation_type)
        raise ValueError(f"Invalid representation type: {representation_type}")

def load_readability_sentences(num_samples: int) -> List[Tuple[str, str]]:
    """
    Load simple vs. complex sentence pairs from the readability explanations dataset.
    """
    max_samples = 500
    if num_samples > max_samples:
        logging.error(
            "Requested %d sentence samples but only %d available",
            num_samples, max_samples
        )
        raise ValueError(f"Data only available for {max_samples} samples, got {num_samples}")

    path = 'datasets/readability/readability_synthetic_data/readability_explanations_500_en.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs: List[Tuple[str, str]] = []
    for key in sorted(data.keys(), key=lambda x: int(x)):
        if len(pairs) >= num_samples:
            break
        entry = data[key]
        simple = entry.get('Simple Summary (With Context)')
        complex = entry.get('Complex Summary')
        if simple is None or complex is None:
            continue
        pairs.append((simple, complex))

    logging.info("Loaded %d sentence pairs", len(pairs))
    return pairs

def load_readability_words(num_samples: int) -> List[Tuple[str, str]]:
    """
    Load simple vs. complex word pairs from the readability words dataset.
    """
    max_samples = 100
    if num_samples > max_samples:
        logging.error(
            "Requested %d word samples but only %d available",
            num_samples, max_samples
        )
        raise ValueError(f"Data only available for {max_samples} samples, got {num_samples}")

    path = 'datasets/readability/readability_synthetic_data/readability_words_100_en.csv'
    simple_words: List[str] = []
    complex_words: List[str] = []

    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            simple_words.append(row['simple'])
            complex_words.append(row['complex'])

    samples: List[Tuple[str, str]] = []
    for s, c in zip(simple_words[:num_samples], complex_words[:num_samples]):
        samples.append((s, c))

    logging.info("Loaded %d word pairs", len(samples))
    return samples


def main() -> None:
    """
    Main function to test the create_readability_vector_training_samples functionality.
    """
    # Test sentences
    logging.info("Testing with sentences representation")
    try:
        sentence_samples = create_readability_vector_training_samples(5, 'sentences')
        logging.info("Generated %d sentence samples", len(sentence_samples))
        for i, (simple, complex) in enumerate(sentence_samples, 1):
            logging.info("Sample %d simple: %s", i, simple)
            logging.info("Sample %d complex: %s", i, complex)
    except Exception as e:
        logging.error("Error with sentences: %s", e)

    # Separator
    logging.info("%s", '-' * 50)

    # Test words
    logging.info("Testing with words representation")
    try:
        word_samples = create_readability_vector_training_samples(5, 'words')
        logging.info("Generated %d word samples", len(word_samples))
        for i, (simple, complex) in enumerate(word_samples, 1):
            logging.info("Sample %d simple: %s", i, simple)
            logging.info("Sample %d complex: %s", i, complex)
    except Exception as e:
        logging.error("Error with words: %s", e)

    # Separator
    logging.info("%s", '-' * 50)

    # Invalid type
    logging.info("Testing invalid representation type")
    try:
        _ = create_readability_vector_training_samples(5, 'invalid_type')
    except ValueError as e:
        logging.info("Expected error caught: %s", e)

    # Too many samples
    logging.info("Testing too many sentences samples")
    try:
        _ = create_readability_vector_training_samples(600, 'sentences')
    except ValueError as e:
        logging.info("Expected error caught: %s", e)

    # Unsupported language
    logging.info("Testing unsupported language")
    try:
        _ = create_readability_vector_training_samples(5, 'sentences', language='fr')
    except ValueError as e:
        logging.info("Expected error caught: %s", e)

    # log the output of the create_readability_vector_training_samples function
    logging.info("Output of the create_readability_vector_training_samples function: %s", create_readability_vector_training_samples(5, 'sentences'))

    logging.info("Output of the create_readability_vector_training_samples function: %s", create_readability_vector_training_samples(5, 'words'))

if __name__ == '__main__':
    main()
