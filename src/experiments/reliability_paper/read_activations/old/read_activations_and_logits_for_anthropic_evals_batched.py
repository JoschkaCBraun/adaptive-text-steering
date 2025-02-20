import os
import json
import logging
import pickle
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_vectors
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al
from src.utils.generate_prompts_for_anthropic_evals_dataset import generate_prompt_and_anwers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def collect_last_activations_and_logits_for_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    layer_nums: List[int],
) -> List[Dict]:
    """
    For a batch of prompts:
      1) Tokenize & run a *single* forward pass with activation hooks.
      2) Record the *last position’s* logits and activations per sample.

    Returns:
      A list of dicts (length = len(prompts)), each with keys:
        {
          "logits": Tensor(shape [vocab_size]),
          "activations": { layer_num: Tensor(shape [hidden_dim]) }
        }
    """
    # 1) Tokenize (with padding) and move to device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)           # (batch_size, seq_len)
    attention_mask = inputs["attention_mask"].to(device) # (batch_size, seq_len)
    batch_size = input_ids.size(0)

    # 2) Forward pass with hooking
    with torch.no_grad(), steering_vectors.record_activations(
        model,
        layer_type="decoder_block",
        layer_nums=layer_nums
    ) as recorded_acts:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.logits: (batch_size, seq_len, vocab_size)

    # 3) Extract the last position's logits (the final token in each prompt)
    #    shape: (batch_size, vocab_size)
    last_pos_logits = outputs.logits[:, -1, :].cpu()

    # 4) Extract the last position's activations from each requested layer
    #    recorded_acts[layer_num] is typically a list of length 1
    #    with shape (batch_size, seq_len, hidden_dim)
    results = []
    for i in range(batch_size):
        # Prepare a dictionary for i-th sample
        res_i = {}
        res_i["logits"] = last_pos_logits[i]  # shape [vocab_size]

        # Extract the last token's activation from each layer
        activations_per_layer = {}
        for ln in layer_nums:
            # For a standard decoder block, recorded_acts[ln][0]:
            #   shape = (batch_size, seq_len, hidden_dim)
            all_acts_for_layer = recorded_acts[ln][0]
            activations_per_layer[ln] = all_acts_for_layer[i, -1, :].cpu()
        res_i["activations"] = activations_per_layer

        results.append(res_i)

    return results


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
    output_dir: str
) -> None:
    """
    Process an entire dataset, iterating over ALL prompt scenarios in `prompt_configurations`.
    Stores results in a pickle file.

    - If scenario is NOT prefilled:
        We do ONE forward pass (per batch) on the prompt and store the final activations/logits.
    - If scenario IS prefilled:
        For each sample, we do TWO forward passes:
          * prompt + matching_answer
          * prompt + non_matching_answer
        Then store each set of last activations/logits under different keys.
    """
    metadata = {
        "dataset_info": {
            "name": dataset_name,
            "type": dataset_type,
            "num_samples": num_samples
        },
        "model_info": {
            "name": model_name,
            "layer_nums": layer_nums
        },
        "scenarios": prompt_configurations
    }

    # scenario_name -> list of sample results (each result is a dict)
    results = {scenario_name: [] for scenario_name in prompt_configurations.keys()}

    # e.g. "anthropic_evals_activations/<dataset_name>/<model_name>_<dataset_name>_activations_10_samples.pkl"
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}/{model_name}_{dataset_name}_activations_and_logits_for_{num_samples}_samples.pkl"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Go scenario by scenario
    for scenario_name, scenario_settings in prompt_configurations.items():
        logging.info(f"[{dataset_name}] Processing scenario: {scenario_name}")

        # Tells us how to do the forward pass logic
        is_prefilled = scenario_settings.get("prefilled", False)

        # Generate N prompts for the scenario
        # returns 3 parallel lists of length num_samples
        prompts, matching_answers, non_matching_answers = generate_prompt_and_anwers(
            dataset_name,
            num_samples=num_samples,
            **scenario_settings
        )

        # Process in batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_prompts = prompts[start_idx:end_idx]
            batch_matching_answers = matching_answers[start_idx:end_idx]
            batch_non_matching_answers = non_matching_answers[start_idx:end_idx]

            if not is_prefilled:
                # ----- NOT PREFILLED: Single forward pass per sample in batch -----
                batch_results = collect_last_activations_and_logits_for_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    device=device,
                    layer_nums=layer_nums
                )

                # Attach scenario-specific info
                for i, br in enumerate(batch_results):
                    results[scenario_name].append({
                        "prompt": batch_prompts[i],
                        "answer_matching_behavior": batch_matching_answers[i],
                        "answer_not_matching_behavior": batch_non_matching_answers[i],
                        "logits": br["logits"],           # shape [vocab_size]
                        "activations": br["activations"]  # dict: layer_num -> [hidden_dim]
                    })

            else:
                # ----- PREFILLED: Two forward passes for each sample -----

                # 1) Construct "prompt + matching_answer"
                batch_prompts_matched = [
                    batch_prompts[i] + batch_matching_answers[i]
                    for i in range(len(batch_prompts))
                ]
                # 2) Collect last-position activations/logits
                batch_results_matched = collect_last_activations_and_logits_for_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts_matched,
                    device=device,
                    layer_nums=layer_nums
                )

                # 3) Construct "prompt + non_matching_answer"
                batch_prompts_non_matched = [
                    batch_prompts[i] + batch_non_matching_answers[i]
                    for i in range(len(batch_prompts))
                ]
                # 4) Collect last-position activations/logits
                batch_results_non_matched = collect_last_activations_and_logits_for_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts_non_matched,
                    device=device,
                    layer_nums=layer_nums
                )

                # 5) Combine each sample into a single dictionary
                for i, prompt in enumerate(batch_prompts):
                    results[scenario_name].append({
                        "prompt": prompt,
                        # We can store the original matching/nonmatching text as well
                        "prefilled_with_matching": {
                            "prefill_text": batch_matching_answers[i],
                            "logits": batch_results_matched[i]["logits"],
                            "activations": batch_results_matched[i]["activations"]
                        },
                        "prefilled_with_nonmatching": {
                            "prefill_text": batch_non_matching_answers[i],
                            "logits": batch_results_non_matched[i]["logits"],
                            "activations": batch_results_non_matched[i]["activations"]
                        }
                    })

            # Save intermediate to file
            intermediate_results = {"metadata": metadata, "results": results}
            with open(output_file, 'wb') as f:
                pickle.dump(intermediate_results, f)

            logging.info(f"  Saved up to sample {end_idx} for scenario '{scenario_name}' in {output_file}")


def main() -> None:
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "anthropic_evals_activations")
    DATASETS_PATH = utils.get_path('DATASETS_PATH')

    # Number of datasets to process
    NUM_DATSETS = 2
    # Number of samples you want per scenario
    NUM_SAMPLES = 8
    BATCH_SIZE = 8

    # 1) Load the prompt configurations
    PROMPT_CONFIGS_PATH = f"{DATASETS_PATH}/anthropic_evals/prompts/prompt_configurations.json"
    with open(PROMPT_CONFIGS_PATH, 'r') as file:
        PROMPT_CONFIGURATIONS = json.load(file)

    device = utils.get_device()
    torch.set_grad_enabled(False)
    logging.info(f"Using device: {device}")

    # 2) Models to process
    models = [
        ('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [13])
    ]

    # 3) Datasets
    datasets = all_datasets_with_type_figure_13_tan_et_al[:NUM_DATSETS]

    # 4) For each model, for each dataset, run all prompt scenarios
    for model_name, model_path, layer_nums in models:
        logging.info(f"Loading model {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
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
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
