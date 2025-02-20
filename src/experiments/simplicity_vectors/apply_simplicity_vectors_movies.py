import os
import sys
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# pylint: disable=wrong-import-position
from config.experiment_config import ExperimentConfig
from src.utils.load_and_get_utils import load_model_and_tokenizer, get_data_dir
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedReviewGenerator:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer, self.model, self.device = load_model_and_tokenizer(config)
        self.model.to(self.device)
        self.model.eval()
        self.simplicity_vector = self._load_simplicity_vector()
        self.prompts = [
            "Write a movie review for [MOVIE]:",
            "Provide your thoughts on the film [MOVIE]:",
            "Describe your experience watching [MOVIE]:"
        ]
        self.sentiment_strengths = [0.08, 0.04, 0.01, 0.0, -0.01, -0.04, -0.08]

    def _load_simplicity_vector(self) -> Dict[str, float]:
        vector_path = os.path.join(get_data_dir(os.getcwd()), 'simplicity_vectors_data', 
                                   'simplicity_vectors_activations', 
                                   'simplicity_explanations_topic_vector_en_500_for_gemma_2b_all.pkl')
        try:
            with open(vector_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading simplicity vector: {e}")
            raise

    def generate_review(self, prompt: str, sentiment_strength: float, max_new_tokens: int = 120) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with self.simplicity_vector.apply(self.model, multiplier=sentiment_strength):
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_reviews(self, movies: List[str]) -> Dict:
        results = {}
        for movie in movies:
            results[movie] = {}
            for prompt_template in self.prompts:
                prompt = prompt_template.replace("[MOVIE]", movie)
                results[movie][prompt] = {}
                for strength in self.sentiment_strengths:
                    try:
                        review = self.generate_review(prompt, strength)
                        results[movie][prompt][str(strength)] = review
                        logger.info(f"Generated review for '{movie}' with strength {strength}")
                    except Exception as e:
                        logger.error(f"Error generating review for '{movie}' with strength {strength}: {e}")
                        results[movie][prompt][str(strength)] = f"Error: {str(e)}"
        return results

def save_results(results: Dict, config: ExperimentConfig):
    output_dir = os.path.join(get_data_dir(os.getcwd()), 'simplicity_vectors_data',
                              'simplicity_vectors_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simplicity_reviews_{config.MODEL_ALIAS}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    metadata = {
        "model_name": config.MODEL_ALIAS,
        "date_generated": timestamp,
        "simplicity_vector": "simplicity_explanations_topic_vector_en_500_for_gemma_2b_all.pkl"
    }
    
    output = {
        "metadata": metadata,
        "results": results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {filepath}")

def main():
    config = ExperimentConfig()
    movies = [
        "The Matrix", "Titanic", "Avatar",
    ]
    movies_long = [
        "The Matrix", "Titanic", "Avatar", "Harry Potter", "Inception",
        "Jurassic Park", "Gladiator", "Star Wars", "Fight Club", "Pulp Fiction",
        "Harry Potter and the Philosopher's Stone",
        "Harry Potter and the Chamber of Secrets",
        "Harry Potter and the Prisoner of Azkaban",
        "Harry Potter and the Goblet of Fire",
        "Harry Potter and the Order of the Phoenix",
        "Harry Potter and the Half-Blood Prince",
        "Harry Potter and the Deathly Hallows – Part 1",
        "Harry Potter and the Deathly Hallows – Part 2",
        "The Lord of the Rings: The Fellowship of the Ring",
        "The Lord of the Rings: The Two Towers",
        "The Lord of the Rings: The Return of the King",
    ]

    generator = SimplifiedReviewGenerator(config)
    results = generator.generate_reviews(movies)
    save_results(results, config)

if __name__ == "__main__":
    main()