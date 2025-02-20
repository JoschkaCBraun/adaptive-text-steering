"""
read_activations_and_logits_for_anthropic_evals_batched_h5py.py
"""

import os
import json
import h5py
from time import time
import logging
import datetime
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import steering_vectors
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt_and_anwers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def collect_last_activations_and_logits_for_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    layer_nums: List[int],
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
        layer_nums=layer_nums
    ) as recorded_acts:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_pos_logits = outputs.logits[:, -1, :].cpu().half()

    results = []
    for i in range(batch_size):
        res_i = {
            "logits": last_pos_logits[i],
            "activations": {
                ln: recorded_acts[ln][0][i, -1, :].cpu().half()
                for ln in layer_nums
            }
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
            
            # For non-prefilled scenarios
            data_group.create_dataset('logits', (0, vocab_size), 
                                    maxshape=(None, vocab_size), 
                                    dtype='float16', chunks=True)
            
            for layer in model_info['layer_nums']:
                data_group.create_dataset(f'activations_layer_{layer}', 
                                        (0, hidden_dim),
                                        maxshape=(None, hidden_dim),
                                        dtype='float16', chunks=True)
            
            # For prefilled scenarios (if needed)
            prefilled_group = data_group.create_group('prefilled')
            for condition in ['matching', 'nonmatching']:
                condition_group = prefilled_group.create_group(condition)
                condition_group.create_dataset('logits', (0, vocab_size),
                                            maxshape=(None, vocab_size),
                                            dtype='float16', chunks=True)
                for layer in model_info['layer_nums']:
                    condition_group.create_dataset(f'activations_layer_{layer}',
                                                (0, hidden_dim),
                                                maxshape=(None, hidden_dim),
                                                dtype='float16', chunks=True)

def append_to_dataset(dataset, data):
    """Append new data to an HDF5 dataset."""
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
    batch_non_matching_answers: List[str],
    is_prefilled: bool
) -> None:
    """Save a batch of results to the HDF5 file."""
    scenario_group = h5_file[f'scenarios/{scenario_name}']
    
    # Save text data
    append_to_dataset(scenario_group['text/prompts'], batch_prompts)
    append_to_dataset(scenario_group['text/matching_answers'], batch_matching_answers)
    append_to_dataset(scenario_group['text/non_matching_answers'], batch_non_matching_answers)
    
    if not is_prefilled:
        # Stack tensors for efficient storage
        logits = torch.stack([br["logits"] for br in batch_results]).numpy()
        append_to_dataset(scenario_group['data/logits'], logits)
        
        # Save activations for each layer
        for layer, acts in batch_results[0]["activations"].items():
            layer_acts = torch.stack([br["activations"][layer] for br in batch_results]).numpy()
            append_to_dataset(scenario_group[f'data/activations_layer_{layer}'], layer_acts)
    else:
        # For prefilled scenarios, we have separate results for matching/nonmatching
        for condition_idx, condition in enumerate(['matching', 'nonmatching']):
            results_key = f"prefilled_with_{condition}"
            
            # Stack logits
            logits = torch.stack([br[results_key]["logits"] for br in batch_results]).numpy()
            append_to_dataset(scenario_group[f'data/prefilled/{condition}/logits'], logits)
            
            # Stack activations for each layer
            for layer in batch_results[0][results_key]["activations"].keys():
                layer_acts = torch.stack([br[results_key]["activations"][layer] 
                                        for br in batch_results]).numpy()
                append_to_dataset(
                    scenario_group[f'data/prefilled/{condition}/activations_layer_{layer}'],
                    layer_acts
                )

def process_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_type: str,
    prompt_configurations: Dict[str, Dict],
    device: torch.device,
    model_name: str,
    layer_nums: List[int],
    num_samples: int,
    batch_size: int,
    output_dir: str,
    num_workers: int = 0
) -> None:
    """Process dataset and save results incrementally to HDF5 using DataLoader."""
    start_time = time()
    
    # Prepare metadata
    dataset_info = {
        "name": dataset_name,
        "type": dataset_type,
        "num_samples": num_samples
    }
    model_info = {
        "name": model_name,
        "layer_nums": layer_nums
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
            is_prefilled = scenario_settings.get("prefilled", False)
            
            # Generate prompts for the scenario
            prompts, matching_answers, non_matching_answers = generate_prompt_and_anwers(
                dataset_name,
                num_samples=num_samples,
                **scenario_settings
            )

            # Create dataset and dataloader
            dataset = PromptDataset(
                prompts=prompts,
                matching_answers=matching_answers,
                non_matching_answers=non_matching_answers,
                is_prefilled=is_prefilled
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # Keep sequential order
                num_workers=num_workers,
                pin_memory=True  # Speeds up transfer to GPU
            )

            # Process batches using DataLoader
            for batch in dataloader:
                if not is_prefilled:
                    batch_results = collect_last_activations_and_logits_for_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch['prompt'],
                        device=device,
                        layer_nums=layer_nums
                    )
                else:
                    # Process prefilled scenarios
                    batch_results_matched = collect_last_activations_and_logits_for_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch['prefilled_matching'],
                        device=device,
                        layer_nums=layer_nums
                    )

                    batch_results_non_matched = collect_last_activations_and_logits_for_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch['prefilled_nonmatching'],
                        device=device,
                        layer_nums=layer_nums
                    )

                    # Combine results
                    batch_results = []
                    for i in range(len(batch['prompt'])):
                        batch_results.append({
                            "prefilled_with_matching": {
                                "prefill_text": batch['matching_answer'][i],
                                "logits": batch_results_matched[i]["logits"],
                                "activations": batch_results_matched[i]["activations"]
                            },
                            "prefilled_with_nonmatching": {
                                "prefill_text": batch['non_matching_answer'][i],
                                "logits": batch_results_non_matched[i]["logits"],
                                "activations": batch_results_non_matched[i]["activations"]
                            }
                        })

                # Save batch results
                save_batch_results(
                    h5_file,
                    scenario_name,
                    batch_results,
                    batch['prompt'],
                    batch['matching_answer'],
                    batch['non_matching_answer'],
                    is_prefilled
                )

                # Update sample count
                h5_file.attrs['num_samples'] += len(batch['prompt'])
                
                # Free memory
                del batch_results
                torch.mps.empty_cache()
                torch.cuda.empty_cache()

    end_time = time()
    total_time = end_time - start_time
    time_per_sample = total_time / num_samples
    logging.info(f"Dataset '{dataset_name}' processing completed:")
    logging.info(f"  Total time: {total_time:.2f} seconds")
    logging.info(f"  Time per sample: {time_per_sample:.2f} seconds")

def main() -> None:
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "anthropic_evals_activations")
    DATASETS_PATH = utils.get_path('DATASETS_PATH')

    NUM_DATSETS = 36
    NUM_SAMPLES = 200
    BATCH_SIZE = 1

    # Load configurations
    PROMPT_CONFIGS_PATH = f"{DATASETS_PATH}/anthropic_evals/prompts/prompt_configurations.json"
    with open(PROMPT_CONFIGS_PATH, 'r') as file:
        PROMPT_CONFIGURATIONS = json.load(file)

    device = utils.get_device()
    torch.set_grad_enabled(False)
    logging.info(f"Using device: {device}")

    models = [
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [13])
    ]

    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATSETS]

    for model_name, model_path, layer_nums in models:
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
                    layer_nums=layer_nums,
                    num_samples=NUM_SAMPLES,
                    batch_size=BATCH_SIZE,
                    output_dir=ACTIVATIONS_PATH
                )
            except Exception as e:
                logging.error(f"Error processing dataset '{dataset_name}': {e}", exc_info=True)
                continue

        del model, tokenizer
        torch.mps.empty_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
