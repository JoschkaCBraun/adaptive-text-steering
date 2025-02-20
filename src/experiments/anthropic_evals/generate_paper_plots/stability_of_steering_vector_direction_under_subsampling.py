#!/usr/bin/env python3
import os
import json
import random
import h5py
import torch
import numpy as np

# Assuming you have these available from src.utils
import src.utils as utils
from src.utils.dataset_names import all_datasets_figure_13

def parse_scenario_base(raw_name: str):
    """
    Given a base scenario name like 'pre_m_instr_m_5shot', strip out all '_m'/'_nm'
    and see which set of tokens remain among {'pre','instr','5shot'}.
    Then return one of:
      'pre', 'instr', '5shot',
      'pre_instr', 'pre_5shot', 'instr_5shot', 'pre_instr_5shot'
    or None if it doesn't match.
    """
    # Remove all occurrences of "_m" or "_nm" (repeatedly, in case they appear multiple times)
    tmp = raw_name
    for _ in range(10):
        tmp = tmp.replace("_m", "")
        tmp = tmp.replace("_nm", "")
    # Now tmp might be something like 'pre_instr_5shot' or 'pre' etc.
    tokens = [t for t in tmp.split("_") if t]  # split and remove empties

    # Ensure tokens are among the expected ones.
    unique_tokens = set(tokens)
    for tok in unique_tokens:
        if tok not in {"pre", "instr", "5shot"}:
            return None  # unrecognized scenario

    # Determine which of the 7 valid combinations we have
    if unique_tokens == {"pre"}:
        return "pre"
    elif unique_tokens == {"instr"}:
        return "instr"
    elif unique_tokens == {"5shot"}:
        return "5shot"
    elif unique_tokens == {"pre", "instr"}:
        return "pre_instr"
    elif unique_tokens == {"pre", "5shot"}:
        return "pre_5shot"
    elif unique_tokens == {"instr", "5shot"}:
        return "instr_5shot"
    elif unique_tokens == {"pre", "instr", "5shot"}:
        return "pre_instr_5shot"
    else:
        return None

def main() -> None:
    # Hyperparameters
    SUBSAMPLE_SIZE = 25
    NUM_REPETITIONS = 100
    # For demonstration we use a subset of datasets.
    NUM_DATASETS = 36

    # Mapping from raw scenario keys to pretty names
    scenario_to_pretty = {
        "pre": "prefilled",
        "instr": "instruction",
        "5shot": "5-shot",
        "pre_instr": "prefilled instruction",
        "pre_5shot": "prefilled 5-shot",
        "instr_5shot": "instruction 5-shot",
        "pre_instr_5shot": "prefilled instruction 5-shot",
    }

    # Set a random seed for reproducibility (optional)
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = utils.get_device()

    # Select your datasets
    datasets = all_datasets_figure_13[:NUM_DATASETS]

    # Base directory where your HDF5 files are stored
    base_dir = os.path.join(utils.get_path("DISK_PATH"), "anthropic_evals_activations_500")

    # We'll accumulate all results into a single JSON structure
    results = []

    for dataset_name in datasets:
        # Construct the HDF5 file path
        h5_filename = f"llama2_7b_chat_{dataset_name}_activations_and_logits_for_500_samples.h5"
        h5_path = os.path.join(base_dir, dataset_name, h5_filename)

        if not os.path.isfile(h5_path):
            print(f"File not found: {h5_path}. Skipping.")
            continue

        print(f"\nProcessing dataset: {dataset_name}")

        # Structure to hold stability stats for each prompt type in this dataset
        dataset_result = {
            "dataset_name": dataset_name,
            "prompt_types": {}
        }

        with h5py.File(h5_path, "r") as f:
            # We expect a top-level "scenarios" group
            if "scenarios" not in f:
                print(f"Skipping {dataset_name}: no 'scenarios' group in HDF5.")
                continue

            scenarios_group = f["scenarios"]
            all_subgroup_names = list(scenarios_group.keys())

            # Instead of only checking for names ending with "_m", we select those that
            # are intended as matching variants (contain "_m" but not "_nm").
            m_subgroups = [s for s in all_subgroup_names if ("_m" in s and "_nm" not in s)]

            for m_name in m_subgroups:
                # Derive the non-matching group by replacing all occurrences of "_m" with "_nm"
                nm_name = m_name.replace("_m", "_nm")
                if nm_name not in all_subgroup_names:
                    print(f"  Skipping scenario '{m_name}': missing counterpart {nm_name}")
                    continue

                # Use the matching group name (or you could use nm_name) to derive the scenario key.
                scenario_key = parse_scenario_base(m_name)
                if scenario_key is None:
                    print(f"  Skipping scenario '{m_name}' - not one of the 7 valid combos.")
                    continue

                grp_m = scenarios_group[m_name]
                grp_nm = scenarios_group[nm_name]

                # Check that both groups have the expected data
                if "data" not in grp_m or "activations_layer_13" not in grp_m["data"]:
                    print(f"  Skipping '{m_name}' in {dataset_name}: no 'data/activations_layer_13'.")
                    continue
                if "data" not in grp_nm or "activations_layer_13" not in grp_nm["data"]:
                    print(f"  Skipping '{nm_name}' in {dataset_name}: no 'data/activations_layer_13'.")
                    continue

                # Load the activation arrays (expected shape: (500, hidden_size))
                matching_acts_np = grp_m["data"]["activations_layer_13"][...]
                nonmatching_acts_np = grp_nm["data"]["activations_layer_13"][...]

                # Convert to float32 tensors on the chosen device
                matching_acts = torch.from_numpy(matching_acts_np).float().to(device)
                non_matching_acts = torch.from_numpy(nonmatching_acts_np).float().to(device)

                # Check that the shapes match
                if matching_acts.shape != non_matching_acts.shape:
                    print(f"  Skipping '{m_name}' in {dataset_name}: shape mismatch between matching vs nonmatching.")
                    continue

                num_pairs = matching_acts.shape[0]
                print(f"  -> scenario_key: {scenario_key} (from '{m_name}') | #pairs={num_pairs}")

                # Collect sub-steering vectors from random subsamples
                sub_vectors = []
                for _ in range(NUM_REPETITIONS):
                    indices = random.sample(range(num_pairs), SUBSAMPLE_SIZE)
                    
                    pos_subset = matching_acts[indices]
                    neg_subset = non_matching_acts[indices]

                    # Convert each row (activation vector) to a 1D tensor
                    pos_list = list(pos_subset.unbind(dim=0))
                    neg_list = list(neg_subset.unbind(dim=0))

                    # Compute the contrastive steering vector
                    steering_vector = utils.compute_contrastive_steering_vector(
                        positive_activations=pos_list,
                        negative_activations=neg_list,
                        device=device
                    )
                    sub_vectors.append(steering_vector)

                # Compute pairwise cosine similarity among the computed steering vectors
                similarities = utils.compute_pairwise_cosine_similarity(sub_vectors, device=device)
                similarities_np = np.array(similarities)
                mean_sim = float(similarities_np.mean())
                std_sim = float(similarities_np.std())

                # Convert raw scenario key to a pretty name using the mapping
                pretty_key = scenario_to_pretty.get(scenario_key, scenario_key)
                dataset_result["prompt_types"][pretty_key] = {
                    "subsample_size": SUBSAMPLE_SIZE,
                    "num_repetitions": NUM_REPETITIONS,
                    "mean_cosine_similarity": mean_sim,
                    "std_cosine_similarity": std_sim
                }

        results.append(dataset_result)

    # Write all results to a single JSON file
    output_json_path = os.path.join(
        utils.get_path("DATA_PATH"),
        "0_paper_plots",
        "intermediate_data",
        "steering_vector_stability_results.json"
    )
    with open(output_json_path, "w") as f_out:
        json.dump({"analysis": results}, f_out, indent=2)

    print(f"\nDone! Results saved in {output_json_path}")

if __name__ == "__main__":
    main()
