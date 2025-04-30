"""
experiment_config.py
This script contains all the experiment hyperparameters that can be adapted by the user.
"""

# Standard library imports
import os
from dotenv import load_dotenv
import logging
from typing import Any, Dict, List

# Third-party imports
from transformers import GenerationConfig, AutoTokenizer
from config.model_config import MODEL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

#pylint: disable=invalid-name
class ExperimentConfig:
    """
    Config class for storing settings and hyperparameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the configuration with optional keyword arguments.
        """
        # Choose experiment between 'baseline', 'prompt_engineering', 'constant_shift',
        # 'factor_scaling' and 'threshold_selection', 'topic_vectors'
        self.HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.VALID_EXPERIMENT_NAMES = ['prompt_engineering', 'topic_vector', 'sentiment_vector']
        self.VALID_BEHAVIOR_TYPES = ['topic', 'toxicity', 'readability', 'sentiment']
        self.TOPIC_VECTOR_EXPERIMENT_NAMES = ['word_steering', 'phrase_steering', 'description_steering', 'summary_steering']
        self.EXPERIMENT_NAME: str = kwargs.get('EXPERIMENT_NAME', 'sentiment_vector')

        # Choose one of the following models from the AutoModelForCausalLM class in the
        # Hugging Face transformers library: 'openelm_270m', 'openelm_450m', 'openelm_1b',
        # 'openelm_3b', 'gemma_2b', 'gemma_7b', 'falcon_7b', 'mistral_7b', 'llama_8b'.
        # The corresponding tokenizer from the AutoTokenizer class is chosen automatically.
        self.MODEL_ALIAS: str = kwargs.get('MODEL_ALIAS', 'llama3_1b')

        self.EXPERIMENT_CONFIG: Dict[str, str] = {
            'experiment_name': self.EXPERIMENT_NAME,
            'model_alias': self.MODEL_ALIAS,
        }

        self.MODEL_CONFIGS: Dict[str, str] = MODEL_CONFIGS

        # dataset used to train the steering vectors
        self.TRAINING_DATASET_NAME: str = kwargs.get('TRAINING_DATSET_NAME', 'newts_train')
        # dataset used to generate summaries
        self.TEST_DATASET_NAME: str = kwargs.get('TEST_DATASET_NAME', 'newts_test')
        # Number of articles for which summaries are generated
        self.NUM_ARTICLES: int = kwargs.get('NUM_ARTICLES', 3)
        # Batch size for data loader. Code is not prepared for batch size > 1!
        self.BATCH_SIZE: int = kwargs.get('BATCH_SIZE', 1)
        # Number of workers for generating summaries
        self.NUM_WORKERS: int = kwargs.get('NUM_WORKERS', 0)
        # Shuffle data loader
        self.SHUFFLE: bool = kwargs.get('SHUFFLE', False)

        self.DATASET_CONFIG: Dict[str, Any] = {
            'training_dataset_name': self.TRAINING_DATASET_NAME,
            'test_dataset_name': self.TEST_DATASET_NAME,
            'num_articles': self.NUM_ARTICLES,
            'batch_size': self.BATCH_SIZE,
            'num_workers': self.NUM_WORKERS,
            'shuffle': self.SHUFFLE,
        }

        # number of words from topic model to use as features
        self.NUM_TOPIC_WORDS: int = kwargs.get('NUM_TOPIC_WORDS', 25)
        self.BEHAVIOR_WORDS_NUM_SAMPLES: int = kwargs.get('BEHAVIOR_WORDS_NUM_SAMPLES', 100)
        # steering strengths for topic vectors
        self.STEERING_STRENGTHS: List[float] = kwargs.get('STEERING_STRENGTHS', [-5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 5])
        # minimum value of phi to consider when extracting topic words
        self.PAIRING_TYPES: List[str] = kwargs.get('PAIRING_TYPES', ['against_random_topic_representation', 'random_string'])
        self.TOPIC_REPRESENTATION_TYPES: List[str] = kwargs.get('TOPIC_REPRESENTATION_TYPES', ['topic_words', 'topic_phrases', 'topic_descriptions', 'topic_summaries'])
        self.TOPIC_REPRESENTATION_TYPE_NUM_SAMPLES: Dict[str, int] = kwargs.get('TOPIC_REPRESENTATION_TYPE_NUM_SAMPLES', {'topic_words': 30625, 'topic_phrases': 1100, 'topic_descriptions': 49, 'topic_summaries': 5000})
        self.TOPIC_IDS: List[int] = [12, 13, 32, 39, 46, 48, 55, 61, 62, 64, 72, 78, 83, 85, 89, 90, 97, 100, 101, 105, 110, 113, 115, 128, 129, 134, 144, 152, 153, 162, 163, 175, 180, 187, 194, 195, 196, 198, 199, 200, 205, 211, 217, 218, 227, 229, 236, 245, 247, 248]
        self.MIN_PHI_VALUE: float = kwargs.get('MIN_PHI_VALUE', 0.001)

        self.TOPICS_CONFIG: Dict[str, Any] = {
            'num_topic_words': self.NUM_TOPIC_WORDS,
            'min_phi_value': self.MIN_PHI_VALUE,
        }

        # minimum of new tokens to generate
        self.MIN_NEW_TOKENS: int = kwargs.get('MIN_NEW_TOKENS', 1)
        # maximum of new tokens to generate
        self.MAX_NEW_TOKENS: int = kwargs.get('MAX_NEW_TOKENS', 150)
        # number of beams for beam search
        self.NUM_BEAMS: int = kwargs.get('NUM_BEAMS', 1)
        # True for sampling, False for greedy decoding
        self.DO_SAMPLE: bool = kwargs.get('DO_SAMPLE', True)
        # nucleus sampling parameter
        self.TOP_P: float = kwargs.get('TOP_P', 0.95)
        # top-k sampling parameter
        self.TOP_K: int = kwargs.get('TOP_K', 50)
        # penalty for immediate repetitions
        # self.REPETITION_PENALTY: float = kwargs.get('REPETITION_PENALTY', 2.0)
        # prevent repeating any 2-gram
        # self.NO_REPEAT_NGRAM_SIZE: int = kwargs.get('NO_REPEAT_NGRAM_SIZE', 3)
        # eos token id
        # self.EOS_TOKEN_ID: int = kwargs.get('EOS_TOKEN_ID', None)
        # pad token id
        # self.PAD_TOKEN_ID: int = kwargs.get('PAD_TOKEN_ID', None)
        # number of return sequences
        # self.NUM_RETURN_SEQUENCES: int = kwargs.get('NUM_RETURN_SEQUENCES', 1)
        # low memory by executing beam search iteratively instead of in parallel
        # self.LOW_MEMORY: bool = kwargs.get('LOW_MEMORY', True)
        # temperature for sampling
        # self.TEMPERATURE: float = kwargs.get('TEMPERATURE', 1.0)
        # start token id for decoder
        # self.DECODER_START_TOKEN_ID: int = kwargs.get('DECODER_START_TOKEN_ID', 0)
        # self.EARLY_STOPPING: bool = kwargs.get('EARLY_STOPPING', True)

        self.GENERATION_CONFIG: Dict[str, Any] = {
            'min_new_tokens': self.MIN_NEW_TOKENS,
            'max_new_tokens': self.MAX_NEW_TOKENS,
            'num_beams': self.NUM_BEAMS,
            'do_sample': self.DO_SAMPLE,
            'top_p': self.TOP_P,
            'top_k': self.TOP_K,
        }

        # choose topic encoding type for steering vector training from
        # zeros, topic_strings and topical_summaries
        self.TOPIC_ENCODING_TYPE: str = kwargs.get('TOPIC_ENCODING_TYPE', 'zeros')
        # whether to include non-matching samples in training data for the steering vectors
        self.INCLUDE_NON_MATCHING: bool = kwargs.get('INCLUDE_NON_MATCHING', True)
        # number of samples to generate for training the steering vector, if non-matching samples
        # are included in the training data. Non-matching samples added to the training data until
        # this number is reached
        self.NUM_SAMPLES: int = kwargs.get('NUM_SAMPLES', 250)

        self.TOPIC_VECTORS_CONFIG: Dict[str, Dict[str, Any]] = {
            'topic_strings': {
                'topic_encoding_type': 'topic_strings',
            },
            'topical_summaries_excluding_non_matching': {
                'topic_encoding_type': 'topical_summaries',
                'include_non_matching': False,
            },
            'topical_summaries_including_non_matching': {
                'topic_encoding_type': 'topical_summaries',
                'include_non_matching': True,
                'num_samples': self.NUM_SAMPLES,
            },
        }

        # focus types for prompts in prompt engineering experiment. tid1_focus means the prompt
        # focuses on the first topic id, no_focus means the prompt does not focus on any topic id,
        # tid2_focus means the prompt focuses on the second topic id
        self.PROMPT_ENGINEERING_CONFIG: Dict[str, Any] = {
            'focus_types': kwargs.get('focus_types', ['tid1_focus', 'no_focus', 'tid2_focus']),
        }

        self.ROUGE_METRICS: list = kwargs.get('ROUGE_METRICS', ['rouge1', 'rouge2', 'rougeL'])

        self.validate()

    #pylint: disable=line-too-long
    def validate(self) -> None:
        """Validate the configuration settings."""
        if self.NUM_ARTICLES <= 0:
            raise ValueError(f"NUM_ARTICLES must be greater than 0, but is {self.NUM_ARTICLES}")
        if self.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be greater than 0, but is {self.BATCH_SIZE}")
        if not 0 < self.MIN_PHI_VALUE < 1:
            raise ValueError("MIN_PHI_VALUE must be between 0 and 1, but is {self.MIN_PHI_VALUE}")
        if self.MIN_NEW_TOKENS <= 0 or self.MAX_NEW_TOKENS <= 0 or self.MIN_NEW_TOKENS >= self.MAX_NEW_TOKENS:
            raise ValueError("MIN_NEW_TOKENS and MAX_NEW_TOKENS must be greater than 0 and MIN_NEW_TOKENS < MAX_NEW_TOKENS, but are {self.MIN_NEW_TOKENS} and {self.MAX_NEW_TOKENS}")
        if self.NUM_BEAMS <= 0:
            raise ValueError("NUM_BEAMS must be greater than 0")
        if not 0 <= self.TOP_P <= 1:
            raise ValueError("TOP_P must be between 0 and 1, but is {self.TOP_P}")
        if self.TOP_K < 0:
            raise ValueError("TOP_K must be non-negative, but is {self.TOP_K}")
        if self.TOPIC_ENCODING_TYPE not in ['zeros', 'topic_strings', 'topical_summaries']:
            raise ValueError("Invalid TOPIC_ENCODING_TYPE: {self.TOPIC_ENCODING_TYPE}")
        if self.TOPIC_ENCODING_TYPE == 'topical_summaries' and self.INCLUDE_NON_MATCHING not in [True, False]:
            raise ValueError("INCLUDE_NON_MATCHING must be either True or False when using topical summaries, but is {self.INCLUDE_NON_MATCHING}")
        if self.TOPIC_ENCODING_TYPE == 'topical_summaries' and self.NUM_SAMPLES <= 0:
            raise ValueError("NUM_SAMPLES must be greater than 0 when using topical summaries, but is {self.NUM_SAMPLES}")
        if self.EXPERIMENT_NAME not in ['baseline', 'prompt_engineering', 'topic_vectors', 'sentiment_vector']:
            raise ValueError("Invalid experiment name: {self.EXPERIMENT_NAME}")
        if self.MODEL_ALIAS not in self.MODEL_CONFIGS:
            raise ValueError("Invalid model alias: {self.MODEL_ALIAS}")
        if self.EXPERIMENT_NAME not in self.VALID_EXPERIMENT_NAMES:
            raise ValueError("Invalid experiment name: {self.EXPERIMENT_NAME}")
        if self.EXPERIMENT_NAME == 'topic_vectors' and self.TOPIC_ENCODING_TYPE not in self.TOPIC_VECTORS_CONFIG:
            raise ValueError("Invalid topic encoding type for topic vectors: {self.TOPIC_ENCODING_TYPE}")
        if self.EXPERIMENT_NAME == 'prompt_engineering' and not self.PROMPT_ENGINEERING_CONFIG['focus_types']:
            raise ValueError("No focus types specified for prompt engineering")
        if self.BATCH_SIZE > 1:
            raise ValueError("Batch size > 1 is not supported")

        logging.info("Configuration settings are valid.")
    #pylint: enable=line-too-long

    def get_generation_config(self, model_alias: str, tokenizer: AutoTokenizer) -> GenerationConfig:
        '''Get the GenerationConfig for the model alias.
        Args:
            model_alias (str): Alias of the model.
        Returns:
            GenerationConfig: GenerationConfig for the model.'''
        if model_alias not in self.MODEL_CONFIGS:
            raise ValueError(f"Model alias {model_alias} not found in config/model_configurations.")

        model_name = self.MODEL_CONFIGS[model_alias]['model_name']
        generation_config = GenerationConfig.from_pretrained(model_name,
                                                             max_new_tokens=self.MAX_NEW_TOKENS,
                                                             min_new_tokens=self.MIN_NEW_TOKENS,
                                                             num_beams=self.NUM_BEAMS,
                                                             do_sample=self.DO_SAMPLE,
                                                             top_p=self.TOP_P,
                                                             top_k=self.TOP_K,
                                                             max_length=None,
                                                             min_length=None,
                                                             pad_token_id=tokenizer.eos_token_id,
                                                             use_cache=True)
        return generation_config

#pylint: enable=invalid-name

if __name__ == "__main__":
    experiment_config = ExperimentConfig()
    experiment_config.validate()
