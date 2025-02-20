"""
generate_prompts_for_anthropic_evals_dataset.py
"""
import os
import json
from typing import List, Optional, Tuple
import src.utils as utils
from src.utils.dataset_names import all_datasets_figure_13_tan_et_al, \
    all_datasets_with_type_figure_13_tan_et_al

def generate_prompt_and_anwers(dataset_name: str, matching: bool, prefilled: bool, num_samples: int,
                               instruction: bool = False, few_shot: bool = False,
                               num_shots: Optional[int] = 0, start_index: int = 0
                               ) -> Tuple[List[str], List[str], List[str]]:
    """Generate prompts for the specified dataset from the Anthropic evaluations dataset
    with given parameters.
    
    Args:
        dataset_name: Name of the dataset to generate prompts for
        matching: Whether to generate prompt for matching or non-matching behavior
        prefilled: Whether to include the answer in the prompt
        num_samples: Number of prompts to generate (max 500)
        instruction: Whether to include the dataset instruction
        few_shot: Whether to include few-shot examples
        num_shots: Number of few-shot examples to include
        start_index: Index to start generating prompts from
    
    Returns:
        List of generated prompts
    """
    dataset_length = 500
    if dataset_name not in all_datasets_figure_13_tan_et_al:
        raise ValueError(f"dataset_name should be one of {all_datasets_figure_13_tan_et_al}")

    if num_samples > dataset_length or num_samples < 1:
        raise ValueError(f"num_samples should be between 1 and {dataset_length}")
    
    if few_shot and num_shots < 1:
        raise ValueError("num_shots should be provided and greater than 0")

    if num_shots:
        if num_shots >= dataset_length or num_shots < 1:
            raise ValueError("num_shots should be between 1 and 499")

    if start_index < 0 or start_index >= dataset_length:
        raise ValueError(f"start_index should be between 0 and {dataset_length-1}")

    total_needed = num_samples + num_shots
    if start_index + total_needed > dataset_length:
        raise ValueError(f"start_index + num_samples + num_shots should be less than {dataset_length}")
   
    dataset_type = get_dataset_type(dataset_name)
    total_samples = load_dataset_samples(dataset_name, dataset_type, total_needed, start_index)
    few_shot_samples = total_samples[:num_shots]
    samples = total_samples[num_shots:]

    prompts = []
    matching_answers = []
    non_matching_answers = []
    for sample in samples:
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(load_dataset_instruction(dataset_name, matching))
            prompt_parts.append("\n\n")
        
        if few_shot:
            for few_shot_sample in few_shot_samples:
                example = format_question(few_shot_sample, dataset_type)
                example += f"{get_answer(few_shot_sample, matching)}"
                prompt_parts.append(example)
                prompt_parts.append("\n\n")
        
        question = format_question(sample, dataset_type)
        if prefilled:
            answer = get_answer(sample, matching)
            if dataset_type == "xrisk":
                answer = answer[:-1] # Remove the bracket at the end
            question += answer
        if not prefilled and dataset_type == "xrisk":
            question += " ("
        prompt_parts.append(question)
        
        prompts.append("".join(prompt_parts))
        matching_answers.append(get_answer(sample, True))
        non_matching_answers.append(get_answer(sample, False))
    
    return prompts, matching_answers, non_matching_answers

def load_dataset_samples(dataset_name: str, dataset_type: str, num_samples: int,
                         start_index: int = 0) -> List[dict]:
    """Load samples from the dataset file."""

    DATASETS_PATH = utils.get_path('DATASETS_PATH')
    file_path = os.path.join(DATASETS_PATH, "anthropic_evals", dataset_type, f"{dataset_name}.jsonl")
    
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue
            elif len(samples) >= num_samples:
                break
            samples.append(json.loads(line))
    if len(samples) != num_samples:
        raise ValueError(f"Could not load {num_samples} samples from dataset: {dataset_name}")

    return samples

def load_dataset_instruction(dataset_name: str, matching: bool) -> str:
    """Load the instruction for a specific dataset from prompts.json."""
    DATASETS_PATH = utils.get_path('DATASETS_PATH')
    with open(os.path.join(DATASETS_PATH, "anthropic_evals", "prompts", "prompts.json"), 'r') as f:
        prompts = json.load(f)
    
    if dataset_name not in prompts:
        raise ValueError(f"No prompts found for dataset: {dataset_name}")
    
    return prompts[dataset_name]["positive_prompt" if matching else "negative_prompt"]

def format_question(sample: dict, dataset_type: str) -> str:
    """Format a single question based on dataset type."""
    question = sample["question"]
    
    if dataset_type == "xrisk":
        return f"Question: {question}\n\nAnswer:"
    else:  # persona
        return f"Question: {question}\n\nChoices:\n - Yes\n - No\n\nAnswer:"

def get_dataset_type(dataset_name: str) -> str:
    """Get the type (xrisk/persona) for a dataset."""
    for name, dataset_type in all_datasets_with_type_figure_13_tan_et_al:
        if name == dataset_name:
            return dataset_type
    raise ValueError(f"Unknown dataset: {dataset_name}")

def get_answer(sample: dict, matching: bool) -> str:
    """Get the appropriate answer based on matching parameter."""
    return sample["answer_matching_behavior"] if matching else sample["answer_not_matching_behavior"]

def main() -> None:
    """Demonstrate the prompt generator with various configurations."""
    # Example dataset and configurations
    dataset_name = "narcissism"
    test_configs = [
        {
            "name": "Basic question only",
            "params": {
                "matching": True,
                "prefilled": False,
                "num_samples": 2,
                "instruction": False,
                "few_shot": False
            }
        },
        {
            "name": "With instruction",
            "params": {
                "matching": True,
                "prefilled": False,
                "num_samples": 2,
                "instruction": True,
                "few_shot": False
            }
        },
        {
            "name": "With prefilled answers",
            "params": {
                "matching": False,
                "prefilled": True,
                "num_samples": 2,
                "instruction": True,
                "few_shot": False
            }
        },
        {
            "name": "With few-shot examples",
            "params": {
                "matching": True,
                "prefilled": False,
                "num_samples": 2,
                "instruction": False,
                "few_shot": True,
                "num_shots": 1
            }
        },
        {
            "name": "Complete configuration",
            "params": {
                "matching": True,
                "prefilled": True,
                "num_samples": 2,
                "instruction": True,
                "few_shot": True,
                "num_shots": 2
            }
        }
    ]

    # Run each configuration and display results
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"Parameters: {config['params']}")
        print('-'*80)
        
        try:
            prompts, _, _ = generate_prompt_and_anwers(dataset_name, **config['params'])
            for i, prompt in enumerate(prompts, 1):
                print(f"\nPrompt {i}:")
                print(prompt)
                print('-'*40)
        except Exception as e:
            print(f"Error generating prompts: {str(e)}")

if __name__ == "__main__":
    main()