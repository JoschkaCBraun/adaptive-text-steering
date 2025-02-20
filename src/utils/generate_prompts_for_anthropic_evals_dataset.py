"""
generate_prompts_for_anthropic_evals_dataset.py

This module generates prompts for the Anthropic evaluations dataset with configurable components:

Components:
1. Base Question (always included)
2. Prefilled Answer (optional)
   - If included, can be matching or non-matching
3. Instructions (optional)
   - If included, can be matching or non-matching
4. Few-shot Examples (optional)
   - If included, specify number of shots and whether they're matching or non-matching

The module allows for flexible combination of these components to generate different types of
prompts for evaluation purposes. Each component can be configured to either match or not match
the desired behavior, allowing for testing of different prompt structures and their effects.
"""
import os
import json
from typing import List, Optional, Tuple
import src.utils as utils
from src.utils.dataset_names import all_datasets_figure_13, \
    all_datasets_with_type_figure_13

def generate_prompt_and_anwers(
    dataset_name: str,
    num_samples: int,
    start_index: int = 0,
    # Core configuration
    prefilled_answer: bool = False,
    instruction: bool = False,
    few_shot: bool = False,
    # Matching behavior configuration
    prefilled_is_matching: Optional[bool] = None,
    instruction_is_matching: Optional[bool] = None,
    few_shot_is_matching: Optional[bool] = None,
    # Few-shot specific configuration
    num_shots: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Generate prompts for the specified dataset with configurable components.
    
    Each prompt can include:
    1. Base question (always included)
    2. Prefilled answer (optional, can be matching or non-matching)
    3. Instructions (optional, can be matching or non-matching)
    4. Few-shot examples (optional, specify count and matching behavior)
    
    Args:
        dataset_name: Name of the dataset to generate prompts for
        num_samples: Number of prompts to generate (max 500)
        start_index: Index to start generating prompts from
        prefilled_answer: Whether to include the answer in the prompt
        instruction: Whether to include the dataset instruction
        few_shot: Whether to include few-shot examples
        prefilled_is_matching: If prefilled_answer is True, whether it should match behavior
        instruction_is_matching: If instruction is True, whether it should match behavior
        few_shot_is_matching: If few_shot is True, whether examples should match behavior
        num_shots: If few_shot is True, number of examples to include
    
    Returns:
        Tuple of (prompts, matching_answers, non_matching_answers)
    """
    dataset_length = 500
    if dataset_name not in all_datasets_figure_13:
        raise ValueError(f"dataset_name should be one of {all_datasets_figure_13}")

    if num_samples > dataset_length or num_samples < 1:
        raise ValueError(f"num_samples should be between 1 and {dataset_length}")
    
    # Validate parameter combinations
    if few_shot:
        if num_shots is None or num_shots < 1:
            raise ValueError("num_shots must be provided and > 0 when few_shot is True")
        if few_shot_is_matching is None:
            raise ValueError("few_shot_is_matching must be specified when few_shot is True")
    else:
        num_shots = 0
        few_shot_is_matching = None

    if prefilled_answer and prefilled_is_matching is None:
        raise ValueError("prefilled_is_matching must be specified when prefilled_answer is True")
        
    if instruction and instruction_is_matching is None:
        raise ValueError("instruction_is_matching must be specified when instruction is True")

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
        # Build prompt components in order
        prompt_parts = []
        
        # 1. Add instruction if requested
        if instruction:
            prompt_parts.append(load_dataset_instruction(dataset_name, instruction_is_matching))
            prompt_parts.append("\n\n")
        
        # 2. Add few-shot examples if requested
        if few_shot:
            for few_shot_sample in few_shot_samples:
                example = format_question(few_shot_sample, dataset_type)
                example += f"{get_answer(few_shot_sample, few_shot_is_matching)}"
                prompt_parts.append(example)
                prompt_parts.append("\n\n")
        
        # 3. Add the main question
        question = format_question(sample, dataset_type)
        
        # 4. Add prefilled answer if requested
        if prefilled_answer:
            answer = get_answer(sample, prefilled_is_matching)
            if dataset_type == "xrisk":
                answer = answer[:-1]  # Remove the bracket at the end
            question += answer
        if not prefilled_answer and dataset_type == "xrisk":
            question += " ("
            
        prompt_parts.append(question)
        
        # Store results
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

def load_dataset_instruction(dataset_name: str, instruction_is_matching: bool) -> str:
    """Load the instruction for a specific dataset from prompts.json."""
    DATASETS_PATH = utils.get_path('DATASETS_PATH')
    with open(os.path.join(DATASETS_PATH, "anthropic_evals", "prompts", "prompts.json"), 'r') as f:
        prompts = json.load(f)
    
    if dataset_name not in prompts:
        raise ValueError(f"No prompts found for dataset: {dataset_name}")
    
    return prompts[dataset_name]["positive_prompt" if instruction_is_matching else "negative_prompt"]

def format_question(sample: dict, dataset_type: str) -> str:
    """Format a single question based on dataset type."""
    question = sample["question"]
    
    if dataset_type == "xrisk":
        return f"Question: {question}\n\nAnswer:"
    else:  # persona
        return f"Question: {question}\n\nChoices:\n - Yes\n - No\n\nAnswer:"

def get_dataset_type(dataset_name: str) -> str:
    """Get the type (xrisk/persona) for a dataset."""
    for name, dataset_type in all_datasets_with_type_figure_13:
        if name == dataset_name:
            return dataset_type
    raise ValueError(f"Unknown dataset: {dataset_name}")

def get_answer(sample: dict, prefilled_is_matching: bool) -> str:
    """Get the appropriate answer based on matching parameter."""
    return sample["answer_matching_behavior"] if prefilled_is_matching else sample["answer_not_matching_behavior"]

def main() -> None:
    """Demonstrate the prompt generator with various configurations."""
    dataset_name = "corrigible-neutral-HHH"
    
    # Test configurations matching the new structure
    test_configs = [
        {
            "name": "only_question",
            "params": {
                "prefilled_answer": False,
                "instruction": False,
                "few_shot": False,
                "num_samples": 1
            }
        },
        {
            "name": "prefilled_matching_answer",
            "params": {
                "prefilled_answer": True,
                "prefilled_is_matching": True,
                "instruction": False,
                "few_shot": False,
                "num_samples": 1
            }
        },
        {
            "name": "matching_instruction",
            "params": {
                "prefilled_answer": False,
                "instruction": True,
                "instruction_is_matching": True,
                "few_shot": False,
                "num_samples": 1
            }
        },
        {
            "name": "matching_5_shot",
            "params": {
                "prefilled_answer": False,
                "instruction": False,
                "few_shot": True,
                "few_shot_is_matching": True,
                "num_shots": 5,
                "num_samples": 1
            }
        },
        {
            "name": "prefilled_matching_answer_matching_instruction_matching_5_shot",
            "params": {
                "prefilled_answer": True,
                "prefilled_is_matching": True,
                "instruction": True,
                "instruction_is_matching": True,
                "few_shot": True,
                "few_shot_is_matching": True,
                "num_shots": 5,
                "num_samples": 1
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
            prompts, matching_answers, non_matching_answers = generate_prompt_and_anwers(
                dataset_name, **config['params']
            )
            for i, (prompt, m_ans, nm_ans) in enumerate(
                zip(prompts, matching_answers, non_matching_answers), 1
            ):
                print(f"\nPrompt {i}:")
                print(prompt)
                print(f"\nMatching answer: {m_ans}")
                print(f"Non-matching answer: {nm_ans}")
                print('-'*40)
        except Exception as e:
            print(f"Error generating prompts: {str(e)}")

if __name__ == "__main__":
    main()