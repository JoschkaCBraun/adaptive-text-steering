'''
text_processing_utils.py

This script contains the TextProcessor class, which is used for processing text, 
such as lemmatizing and stemming words.
'''

# Standard library imports
import string
import logging
from typing import List, Set

# Third-party imports
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords
import spacy
from spacy.cli import download as spacy_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPACY_MODEL_NAME = 'en_core_web_sm'

class TextProcessor:
    """
    Class for processing text, such as lemmatizing and stemming words.
    Attempts to automatically download the required spaCy model if not found.
    """
    def __init__(self):
        """Initializes necessary resources."""
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        self.lemmatizer = self._load_spacy_model_with_download(SPACY_MODEL_NAME)
        self.stemmer = SnowballStemmer(language='english')
        self.stopwords = set(nltk_stopwords.words('english'))
        self.punctuation = set(string.punctuation)


    def _load_spacy_model_with_download(self, model_name: str):
        """Loads a spaCy model, attempting download if it fails."""
        try:
            # Try loading the model normally
            logger.debug(f"Attempting to load spaCy model '{model_name}'...")
            lemmatizer = spacy.load(model_name, disable=["parser", "ner"])
            logger.info(f"Successfully loaded spaCy model '{model_name}'.")
            return lemmatizer
        except OSError:
            # Catch the error if the model is not found
            logger.warning(f"SpaCy model '{model_name}' not found.")
            if spacy_download:
                logger.info(f"Attempting to download '{model_name}' automatically...")
                try:
                    spacy_download(model_name)
                    logger.info(f"Successfully downloaded '{model_name}'. Retrying load...")
                    # Retry loading after successful download
                    lemmatizer = spacy.load(model_name, disable=["parser", "ner"])
                    logger.info(f"Successfully loaded spaCy model '{model_name}' after download.")
                    return lemmatizer
                except Exception as e:
                    logger.error(f"Failed to automatically download or load '{model_name}' after download attempt: {e}", exc_info=True)
                    # Raise a runtime error to indicate the setup failed
                    raise RuntimeError(f"Failed to automatically download or load spaCy model '{model_name}'. Please install it manually (e.g., 'python -m spacy download {model_name}') and retry.") from e
            else:
                 logger.error(f"Cannot attempt automatic download because spacy download function was not found.")
                 raise RuntimeError(f"SpaCy model '{model_name}' not found and automatic download failed. Please install it manually (e.g., 'python -m spacy download {model_name}') and retry.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading spaCy model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed load spaCy model '{model_name}'. Please check your spaCy installation and model.") from e

    def process_text(self, text: str, method: str) -> Set[str]:
        """
        Processes text by lemmatizing or stemming it and removing stopwords.

        :param text: The text to process.
        :param method: The method to use for processing the text, either 'lemmatize' or 'stem'.
        :return: A set of processed words.
        """
        text.lower()
        processed_words = set()
        if method == 'lemmatize':
            lemmatized_text = self.lemmatizer(text)
            processed_words = {token.lemma_ for token in lemmatized_text if not 
                                (token.is_punct or token.is_space or token.text in self.stopwords)}
        elif method == 'stem':
            words = text.split()
            processed_words = {self.stemmer.stem(word) for word in words if word not in self.stopwords}

        else :
            logger.error('Invalid method for proces_text: %s.'\
                        'Choose between "lemmatize" and "stem".', method)
        return processed_words

    def get_word_variations(self, word: str) -> List[str]:
        """
        Generates variations of a word, including different capitalizations and spaces.
        
        :param word: The word for which to generate variations.
        :return: A list of word variations.
        """
        lemmatized_word = self.lemmatizer(word)[0].lemma_ if len(self.lemmatizer(word)) > 0 else word
        stemmed_word = self.stemmer.stem(word)
        variations = {word, lemmatized_word, stemmed_word}
        variations_capitalization = set()
        for word in variations:
            variations_capitalization.add(word.lower())
            variations_capitalization.add(word.capitalize())
        variations_spaces = set()
        for word in variations_capitalization:
            variations_spaces.add(word)
            variations_spaces.add(' ' + word)
            variations_spaces.add(word + ' ')
            variations_spaces.add(' ' + word + ' ')
        word_variations = [word for word in variations_capitalization if word not in self.stopwords]
        return word_variations
