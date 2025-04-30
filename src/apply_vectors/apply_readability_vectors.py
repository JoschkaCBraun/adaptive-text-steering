'''
apply_sentiment_vectors.py

This script applies sentiment vectors trained on one language to a movie review generation task in another language.
It tests multiple combinations of sentiment vectors and prompts in different languages, with varying sentiment strengths.
'''

# Standard library imports
import os
import sys
import json
import logging
from typing import Dict, List
from dataclasses import dataclass, asdict

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
# pylint: disable=wrong-import-position
from config.experiment_config import ExperimentConfig
from src.experiments.language_generalisation.get_sentiment_vectors import get_sentiment_vector
from utils.load_topic_lda import load_model_and_tokenizer, get_data_dir
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
@dataclass
class ReviewResult:
    '''Dataclass to store the result of a single review generation.'''
    movie: str
    prompt_language: str
    vector_language: str
    sentiment_strength: float
    review: str

class SentimentReviewGenerator:
    '''Class to generate movie reviews using sentiment vectors.'''

    def __init__(self, config: ExperimentConfig, languages: List[str], num_samples: int):
        '''Initialize the SentimentReviewGenerator with given configuration.'''
        self.config = config
        self.languages = languages
        self.num_samples = num_samples
        self.tokenizer, self.model, self.device = load_model_and_tokenizer(config)
        self.model.to(self.device)
        self.model.eval()
        self.sentiment_vectors = self._load_sentiment_vectors()

    def _load_sentiment_vectors(self) -> Dict[str, Dict[str, float]]:
        '''Load or train sentiment vectors for specified languages.'''
        sentiment_vectors = {}
        for lang in self.languages:
            vector = get_sentiment_vector(self.config, lang, self.num_samples)
            if vector is None:
                logger.error(f"Failed to load or train sentiment vector for {lang}")
            else:
                sentiment_vectors[lang] = vector
        return sentiment_vectors

    @staticmethod
    def _generate_prompt(movie: str, language: str) -> str:
        '''Generate a prompt for movie review in the specified language.'''
        prompts = {
            "en": f"Write a movie review for {movie}:",
            "fr": f"Écrivez une critique du film {movie}:",
            "de": f"Schreiben Sie eine Filmkritik für {movie}:"
        }
        return prompts.get(language, prompts["en"])

    def generate_review(self, prompt: str, sentiment_vector: Dict[str, float], 
                        sentiment_strength: float, max_new_tokens: int = 120) -> str:
        '''Generate a movie review using the model with applied sentiment vector.'''
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with sentiment_vector.apply(self.model, multiplier=sentiment_strength):
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_reviews(self, movies: List[str]) -> List[ReviewResult]:
        '''Generate reviews for given movies with different language and sentiment combinations.'''
        results = []
        for movie in movies:
            for prompt_lang in self.languages:
                prompt = self._generate_prompt(movie, prompt_lang)
                for vector_lang, vector in self.sentiment_vectors.items():
                    for sentiment_strength in [0.6, 0.2, 0.0, -0.2, -0.6]:
                        try:
                            review = self.generate_review(prompt, vector, sentiment_strength)
                            result = ReviewResult(
                                movie=movie,
                                prompt_language=prompt_lang,
                                vector_language=vector_lang,
                                sentiment_strength=sentiment_strength,
                                review=review
                            )
                            results.append(result)
                            logger.info(f"Generated review for {movie} in {prompt_lang} using {vector_lang} vector with sentiment strength {sentiment_strength}")
                        except Exception as e:
                            logger.error(f"Error generating review for {movie} in {prompt_lang} using {vector_lang} vector with sentiment strength {sentiment_strength}: {e}")
        return results

def save_results(results: List[ReviewResult], config: ExperimentConfig, num_samples: int):
    '''Save generated reviews and metadata to a JSON file.'''
    output_dir = os.path.join(get_data_dir(os.getcwd()), 'sentiment_vectors_data',
                              'sentiment_vectors_results')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"results_{num_samples}_{config.MODEL_ALIAS}_{len(results)}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump([asdict(result) for result in results], f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {filepath}")

def main():
    config = ExperimentConfig()
    languages = ["en", "fr", "de"]
    num_samples = 250 
    movies = [
        "The Matrix",
        "Titanic",
        "Avatar",
        "Harry Potter",
        "Inception",
        "Jurassic Park",
        "Gladiator",
        "Star Wars",
        "Fight Club",
        "Pulp Fiction",
    ]
    movies_long = [
            "Harry Potter and the Philosopher's Stone",
            "Harry Potter and the Chamber of Secrets",
            "Harry Potter and the Prisoner of Azkaban",
            "Harry Potter and the Goblet of Fire",
            "Harry Potter and the Order of the Phoenix",
            "Harry Potter and the Half-Blood Prince",
            "Harry Potter and the Deathly Hallows – Part 1",
            "Harry Potter and the Deathly Hallows – Part 2",
            "Titanic",
            "Avatar",
            "The Lord of the Rings: The Fellowship of the Ring",
            "The Lord of the Rings: The Two Towers",
            "The Lord of the Rings: The Return of the King",
            "Inception",
            "The Matrix",
            "Jurassic Park",
            ]
    generator = SentimentReviewGenerator(config, languages, num_samples)
    results = generator.generate_reviews(movies)
    save_results(results, config, num_samples)
if __name__ == "__main__":
    main()
# pylint: enable=logging-fstring-interpolation
