import os
import sys
import json
import pickle
import logging
import csv
from datetime import datetime
from typing import Dict, List

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# pylint: disable=wrong-import-position
from config.experiment_config import ExperimentConfig
from utils.load_topic_lda import load_model_and_tokenizer, get_data_dir
# pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedQuestionAnswerer:
    def __init__(self, config: ExperimentConfig, steering_strengths: List[float]):
        self.config = config
        self.tokenizer, self.model, self.device = load_model_and_tokenizer(config)
        self.model.to(self.device)
        self.model.eval()
        self.simplicity_vector = self._load_simplicity_vector()
        self.steering_strengths = steering_strengths

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

    def generate_answer(self, question: str, steering_strength: float, max_new_tokens: int) -> str:
        input_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)
        
        with self.simplicity_vector.apply(self.model, multiplier=steering_strength):
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_answers(self, questions: List[str], max_new_tokens: int) -> Dict:
        results = {}
        for i, question in enumerate(questions):
            results[str(i)] = {"question": question, "answers": {}}
            for strength in self.steering_strengths:
                try:
                    answer = self.generate_answer(question, strength, max_new_tokens)
                    results[str(i)]["answers"][str(strength)] = answer
                    logger.info(f"Generated answer for question {i+1} with strength {strength}")
                except Exception as e:
                    logger.error(f"Error generating answer for question {i+1} with strength {strength}: {e}")
                    results[str(i)]["answers"][str(strength)] = f"Error: {str(e)}"
        return results

def load_questions(file_path: str, num_questions: int) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row[0] for row in reader][:num_questions]

def save_results(results: Dict, config: ExperimentConfig, num_questions: int, 
                 max_new_tokens: int, steering_strengths: List[float]):
    output_dir = os.path.join(get_data_dir(os.getcwd()), 'simplicity_vectors_data',
                              'simplicity_vectors_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simplicity_answers_{num_questions}_en_questions_{config.MODEL_ALIAS}_"\
               f"{min(steering_strengths)}_{max(steering_strengths)}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    metadata = {
        "model_name": config.MODEL_ALIAS,
        "date_generated": timestamp,
        "simplicity_vector": "simplicity_explanations_topic_vector_en_500_for_gemma_2b_all.pkl",
        "num_questions": num_questions,
        "max_tokens": max_new_tokens,
        "steering_strengths": steering_strengths
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
    num_questions = 3  # Set the number of questions to generate answers for
    max_new_tokens = 120  # Set the maximum number of tokens for generated answers
    steering_strengths = [0.1, 0.05, 0.02, 0.0, -0.02, -0.05, -0.1]  # Set the steering strengths

    questions_file = 'data/datasets/prompts/500_explanation_questions.csv'
    questions = load_questions(questions_file, num_questions)

    answerer = SimplifiedQuestionAnswerer(config, steering_strengths)
    results = answerer.generate_answers(questions, max_new_tokens)
    save_results(results, config, num_questions, max_new_tokens, steering_strengths)

if __name__ == "__main__":
    main()