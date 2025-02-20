"""
read_activations_and_logits_for_anthropic_evals.py
"""

import os
import json
from time import time
import logging
import datetime
from typing import Dict, List

import h5py
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import steering_vectors
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt_and_anwers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

START_INDEX = 195
NUM_DATSETS = 36
NUM_SAMPLES = 300
BATCH_SIZE = 1

class PromptDataset(Dataset):
    """Dataset for holding prompts and their corresponding answers."""
    def __init__(
        self,
        prompts: List[str],
        matching_answers: List[str],
        non_matching_answers: List[str],
        is_prefilled: bool
    ):
        self.prompts = prompts
        self.matching_answers = matching_answers
        self.non_matching_answers = non_matching_answers
        self.is_prefilled = is_prefilled

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = {
            'prompt': self.prompts[idx],
            'matching_answer': self.matching_answers[idx],
            'non_matching_answer': self.non_matching_answers[idx]
        }
        
        if self.is_prefilled:
            item.update({
                'prefilled_matching': self.prompts[idx] + self.matching_answers[idx],
                'prefilled_nonmatching': self.prompts[idx] + self.non_matching_answers[idx]
            })
            
        return item
    
def clear_cache(device):
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    elif str(device).startswith('mps'):
        torch.mps.empty_cache()
    else:
        raise ValueError(f"Unknown device type: {device}")

def collect_last_activations_and_logits_for_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    layer_num: int,  # Single layer number instead of list
) -> List[Dict]:
    """
    For a batch of prompts:
      1) Tokenize & run a single forward pass with activation hooks.
      2) Record the last position's logits and activations per sample.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    batch_size = input_ids.size(0)

    with torch.no_grad(), steering_vectors.record_activations(
        model,
        layer_type="decoder_block",
        layer_nums=[layer_num]  # Convert single number to list for record_activations
    ) as recorded_acts:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_pos_logits = outputs.logits[:, -1, :].cpu()

    results = []
    for i in range(batch_size):
        res_i = {
            "logits": last_pos_logits[i],
            "activations": recorded_acts[layer_num][0][i, -1, :].cpu()
        }
        results.append(res_i)

    return results

def initialize_h5_file(
    file_path: str,
    model_name: str,
    dataset_info: Dict,
    model_info: Dict,
    scenarios: Dict,
    vocab_size: int,
    hidden_dim: int
) -> None:
    """Initialize HDF5 file structure for storing activations and logits."""
    with h5py.File(file_path, 'w') as f:
        # Store metadata as attributes
        f.attrs['model_name'] = model_name
        f.attrs['dataset_name'] = dataset_info['name']
        f.attrs['dataset_type'] = dataset_info['type']
        f.attrs['creation_date'] = str(datetime.datetime.now())
        f.attrs['num_samples'] = 0
        
        # Store full metadata as JSON
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['dataset_info'] = json.dumps(dataset_info)
        metadata_group.attrs['model_info'] = json.dumps(model_info)
        metadata_group.attrs['scenarios'] = json.dumps(scenarios)

        # Create groups for each scenario
        for scenario_name in scenarios.keys():
            scenario_group = f.create_group(f'scenarios/{scenario_name}')
            
            # Create groups for storing prompts and answers as string datasets
            text_group = scenario_group.create_group('text')
            text_group.create_dataset('prompts', (0,), dtype=h5py.special_dtype(vlen=str), 
                                    maxshape=(None,), chunks=True)
            text_group.create_dataset('matching_answers', (0,), dtype=h5py.special_dtype(vlen=str), 
                                    maxshape=(None,), chunks=True)
            text_group.create_dataset('non_matching_answers', (0,), dtype=h5py.special_dtype(vlen=str), 
                                    maxshape=(None,), chunks=True)

            # Create datasets for logits and activations
            data_group = scenario_group.create_group('data')
            data_group.create_dataset('logits', (0, vocab_size), 
                                    maxshape=(None, vocab_size), 
                                    dtype='float32', chunks=True)
            data_group.create_dataset(f'activations_layer_{model_info["layer_num"]}', 
                                    (0, hidden_dim),
                                    maxshape=(None, hidden_dim),
                                    dtype='float32', chunks=True)

def append_to_dataset(dataset, data) -> None:
    """Append new data to an HDF5 dataset."""
    # If data is a NumPy array of strings, convert it to a list and sanitize.
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.str_):
            data = data.tolist()
            data = [d.replace('\x00', '') for d in data]
        # Otherwise, if numeric, skip sanitation.
    # If data is a list, check if its elements are strings and sanitize if so.
    elif isinstance(data, list):
        if data and isinstance(data[0], str):
            data = [d.replace('\x00', '') for d in data]
    current_size = dataset.shape[0]
    new_size = current_size + len(data)
    dataset.resize(new_size, axis=0)
    dataset[current_size:new_size] = data

def save_batch_results(
    h5_file: h5py.File,
    scenario_name: str,
    batch_results: List[Dict],
    batch_prompts: List[str],
    batch_matching_answers: List[str],
    batch_non_matching_answers: List[str]
) -> None:
    """Save a batch of results to the HDF5 file."""
    scenario_group = h5_file[f'scenarios/{scenario_name}']
    
    # Save text data
    append_to_dataset(scenario_group['text/prompts'], batch_prompts)
    append_to_dataset(scenario_group['text/matching_answers'], batch_matching_answers)
    append_to_dataset(scenario_group['text/non_matching_answers'], batch_non_matching_answers)
    
    # Stack tensors for efficient storage
    logits = torch.stack([br["logits"] for br in batch_results]).numpy()
    activations = torch.stack([br["activations"] for br in batch_results]).numpy()
    
    append_to_dataset(scenario_group['data/logits'], logits)
    append_to_dataset(scenario_group['data/activations_layer_13'], activations)

def process_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_type: str,
    prompt_configurations: Dict[str, Dict],
    device: torch.device,
    model_name: str,
    layer_num: int,
    num_samples: int,
    batch_size: int,
    output_dir: str,
    num_workers: int = 0
) -> None:
    """Process dataset and save results incrementally to HDF5."""
    start_time = time()
    
    # Prepare metadata
    dataset_info = {
        "name": dataset_name,
        "type": dataset_type,
        "num_samples": num_samples
    }
    model_info = {
        "name": model_name,
        "layer_num": layer_num  # Store single layer number in metadata
    }

    # Setup output file
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}/{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.h5"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize H5 file
    initialize_h5_file(
        output_file,
        model_name,
        dataset_info,
        model_info,
        prompt_configurations,
        vocab_size=model.config.vocab_size,
        hidden_dim=model.config.hidden_size
    )

    # Process scenarios
    with h5py.File(output_file, 'a') as h5_file:
        for scenario_name, scenario_settings in prompt_configurations.items():

            generation_settings = {k: v for k, v in scenario_settings.items() if k != 'long_name'}

            # Generate prompts for the scenario
            prompts, matching_answers, non_matching_answers = generate_prompt_and_anwers(
                dataset_name,
                num_samples=num_samples,
                start_index=START_INDEX,
                **generation_settings
            )

            # Create dataset and dataloader
            dataset = PromptDataset(
                prompts=prompts,
                matching_answers=matching_answers,
                non_matching_answers=non_matching_answers,
                is_prefilled=False
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            # Process batches
            for batch in dataloader:
                batch_results = collect_last_activations_and_logits_for_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch['prompt'],
                    device=device,
                    layer_num=layer_num  # Pass the layer number
                )

                # Save batch results
                save_batch_results(
                    h5_file,
                    scenario_name,
                    batch_results,
                    batch['prompt'],
                    batch['matching_answer'],
                    batch['non_matching_answer']
                )

                # Update sample count
                h5_file.attrs['num_samples'] += len(batch['prompt'])
                
                # Free memory
                del batch_results
                clear_cache(device)

    end_time = time()
    total_time = end_time - start_time
    time_per_sample = total_time / num_samples
    logging.info(f"Dataset '{dataset_name}' processing completed:")
    logging.info(f"  Total time: {total_time:.2f} seconds")
    logging.info(f"  Time per sample: {time_per_sample:.2f} seconds")

def main() -> None:
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "anthropic_evals_activations_300")
    DATASETS_PATH = utils.get_path('DATASETS_PATH')

    # Load configurations
    PROMPT_CONFIGS_PATH = f"{DATASETS_PATH}/anthropic_evals/prompts/prompt_configurations.json"
    with open(PROMPT_CONFIGS_PATH, 'r') as file:
        PROMPT_CONFIGURATIONS = json.load(file)

    device = utils.get_device()
    torch.set_grad_enabled(False)
    logging.info(f"Using device: {device}")

    models = [
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', 13)
    ]

    datasets = all_datasets_with_type_figure_13[:NUM_DATSETS]
    datasets = [("corrigible-less-HHH", "xrisk")]

    for model_name, model_path, layer_num in models:
        logging.info(f"Loading model {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(device)
    
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info(f"Set padding token to EOS token: {tokenizer.pad_token}")

        for dataset_name, dataset_type in datasets:
            try:
                logging.info(f"Processing dataset '{dataset_name}' ({dataset_type})")
                process_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    prompt_configurations=PROMPT_CONFIGURATIONS,
                    device=device,
                    model_name=model_name,
                    layer_num=layer_num,
                    num_samples=NUM_SAMPLES,
                    batch_size=BATCH_SIZE,
                    output_dir=ACTIVATIONS_PATH
                )
            except Exception as e:
                logging.error(f"Error processing dataset '{dataset_name}': {e}", exc_info=True)
                continue

        del model, tokenizer
        clear_cache(device)


if __name__ == "__main__":
    main()
