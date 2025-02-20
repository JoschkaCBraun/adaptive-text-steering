# File: glove_binary_handler.py

import numpy as np
from tqdm import tqdm
import os
import logging
from utils.load_and_get_utils import get_data_dir

class GloveBinaryHandler:
    """
    A handler for loading and working with GloVe word embeddings in binary format.
    
    This class provides methods to load GloVe embeddings, retrieve word vectors,
    and find similar words based on vector similarity.
    """

    VALID_DATASETS = ['6B.50d', '6B.100d', '6B.200d', '6B.300d', '840B.300d']

    def __init__(self, dataset='6B.300d', start_path='.'):
        """
        Initialize the GloveBinaryHandler.

        Args:
            dataset (str): The name of the GloVe dataset to load. Default is '6B.300d'.
            start_path (str): The starting path to search for the data directory. Default is current directory.
        
        Raises:
            ValueError: If the specified dataset is not valid or if the data directory is not found.
        """
        self.word_to_index = {}
        self.vectors = None
        self.dataset = dataset

        if dataset not in self.VALID_DATASETS:
            logging.error(f"Invalid dataset '{dataset}'. Valid options are: {', '.join(self.VALID_DATASETS)}")
            raise ValueError(f"Invalid dataset '{dataset}'. Valid options are: {', '.join(self.VALID_DATASETS)}")

        data_dir = get_data_dir(start_path)
        if data_dir is None:
            logging.error("Data directory not found.")
            raise ValueError("Data directory not found.")

        self.binary_dir = os.path.join(data_dir, 'datasets', 'glove', 'glove_binary')
        self.load_embeddings()

    def load_embeddings(self):
        """
        Load the GloVe embeddings from binary files.

        Raises:
            FileNotFoundError: If the required binary files are not found.
        """
        logging.info(f"Loading GloVe embeddings ({self.dataset})...")
        words_file = os.path.join(self.binary_dir, f"glove.{self.dataset}_words.npy")
        vectors_file = os.path.join(self.binary_dir, f"glove.{self.dataset}_vectors.npy")

        if not os.path.exists(words_file) or not os.path.exists(vectors_file):
            logging.error(f"Binary files for dataset '{self.dataset}' not found in {self.binary_dir}")
            raise FileNotFoundError(f"Binary files for dataset '{self.dataset}' not found in {self.binary_dir}")

        try:
            words = np.load(words_file, allow_pickle=True)
            self.vectors = np.load(vectors_file, mmap_mode='r')
        except Exception as e:
            logging.error(f"Error loading binary files: {str(e)}")
            raise

        self.word_to_index = {word: i for i, word in enumerate(tqdm(words, desc="Building index"))}
        logging.info(f"Loaded {len(words)} words with dimension {self.vectors.shape[1]}")

    def get_vector(self, word):
        """
        Get the vector for a given word.

        Args:
            word (str): The word to get the vector for.

        Returns:
            numpy.ndarray or None: The vector for the word, or None if the word is not in the vocabulary.
        """
        if word in self.word_to_index:
            return self.vectors[self.word_to_index[word]]
        logging.info(f"Word '{word}' not found in the vocabulary.")
        return None

    def has_vector(self, word):
        """
        Check if a word has a vector in the embeddings.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word has a vector, False otherwise.
        """
        return word in self.word_to_index

    def find_most_similar(self, query_vector, n=5):
        """
        Find the most similar words to a given query vector.

        Args:
            query_vector (numpy.ndarray): The query vector to compare against.
            n (int): The number of similar words to return. Default is 5.

        Returns:
            list of tuples: A list of (word, similarity) pairs for the most similar words.
        """
        dot_products = np.dot(self.vectors, query_vector)
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        similarities = dot_products / (vector_norms * query_norm)

        top_indices = np.argsort(similarities)[-n:][::-1]

        words = list(self.word_to_index.keys())
        return [(words[i], similarities[i]) for i in top_indices]

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')