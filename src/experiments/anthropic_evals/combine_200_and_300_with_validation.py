#!/usr/bin/env python3
import os
import h5py
import numpy as np
import datetime
import json
import logging

from src.utils.dataset_names import all_datasets_with_type_figure_13

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def decode_string_array(arr):
    """
    Convert an array of bytes or strings into a list of strings.
    """
    return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr.tolist()]


def combine_files(file_path_200, file_path_300, output_path):
    """
    Combines a 200-sample HDF5 file and a 300-sample HDF5 file into a new 500-sample file.
    
    Raises an error and stops if:
      - Either file is missing.
      - The file-level attributes (model_name, dataset_name, dataset_type) do not match.
      - The internal HDF5 structure or dimensions do not match.
      - The number of samples per scenario is not as expected.
    """
    logging.info(f"Combining:\n  {file_path_200}\n  {file_path_300}\n=> {output_path}")
    
    # Open the two source files in read mode
    with h5py.File(file_path_200, 'r') as f200, h5py.File(file_path_300, 'r') as f300:
        # --- Check file-level attributes ---
        for attr in ['model_name', 'dataset_name', 'dataset_type']:
            val200 = f200.attrs.get(attr)
            val300 = f300.attrs.get(attr)
            if val200 != val300:
                raise ValueError(f"Mismatch in attribute '{attr}': {val200} vs {val300}")

        # --- Verify sample counts per scenario and overall totals ---
        # For the 200-sample file:
        scenarios_200 = list(f200['scenarios'].keys())
        n_scenarios_200 = len(scenarios_200)
        first_scenario = scenarios_200[0]
        samples_per_scenario_200 = f200['scenarios'][first_scenario]['data/activations_layer_13'].shape[0]
        if samples_per_scenario_200 != 200:
            raise ValueError(
                f"Expected each scenario in first file to have 200 samples, but got {samples_per_scenario_200} in scenario '{first_scenario}'"
            )
        expected_total_200 = n_scenarios_200 * 200
        if int(f200.attrs.get('num_samples')) != expected_total_200:
            raise ValueError(f"Expected {expected_total_200} total samples in first file but got {f200.attrs.get('num_samples')}")

        # For the 300-sample file:
        scenarios_300 = list(f300['scenarios'].keys())
        n_scenarios_300 = len(scenarios_300)
        first_scenario_300 = scenarios_300[0]
        samples_per_scenario_300 = f300['scenarios'][first_scenario_300]['data/activations_layer_13'].shape[0]
        if samples_per_scenario_300 != 300:
            raise ValueError(
                f"Expected each scenario in second file to have 300 samples, but got {samples_per_scenario_300} in scenario '{first_scenario_300}'"
            )
        expected_total_300 = n_scenarios_300 * 300
        if int(f300.attrs.get('num_samples')) != expected_total_300:
            raise ValueError(f"Expected {expected_total_300} total samples in second file but got {f300.attrs.get('num_samples')}")

        # --- Check for required groups ---
        for grp in ['metadata', 'scenarios']:
            if grp not in f200 or grp not in f300:
                raise ValueError(f"Missing required group '{grp}' in one of the files.")

        # --- Check detailed metadata ---
        # For the detailed metadata, we ignore the "num_samples" field inside dataset_info
        for key in ['dataset_info', 'model_info', 'scenarios']:
            meta200 = f200['metadata'].attrs.get(key)
            meta300 = f300['metadata'].attrs.get(key)
            # Load the JSON strings
            meta200_json = json.loads(meta200)
            meta300_json = json.loads(meta300)
            if key == 'dataset_info':
                # Remove the 'num_samples' key (if present) before comparing
                meta200_json.pop('num_samples', None)
                meta300_json.pop('num_samples', None)
            if meta200_json != meta300_json:
                raise ValueError(f"Mismatch in detailed metadata attribute '{key}': {meta200_json} vs {meta300_json}")

        # --- Process each scenario ---
        combined_scenarios = {}
        if set(scenarios_200) != set(scenarios_300):
            raise ValueError("Mismatch in scenario keys between the two files.")
        
        # For each scenario, load data and text, verify structure, and concatenate
        for scenario in scenarios_200:
            combined_scenarios[scenario] = {}
            grp200 = f200['scenarios'][scenario]
            grp300 = f300['scenarios'][scenario]

            # Check that both have the 'data' and 'text' subgroups
            for subgrp in ['data', 'text']:
                if subgrp not in grp200 or subgrp not in grp300:
                    raise ValueError(f"Missing subgroup '{subgrp}' in scenario '{scenario}' in one of the files.")

            # --- Data group: activations and logits ---
            data200 = grp200['data']
            data300 = grp300['data']

            # Check that expected datasets exist
            for dkey in ['activations_layer_13', 'logits']:
                if dkey not in data200 or dkey not in data300:
                    raise ValueError(f"Missing dataset '{dkey}' in scenario '{scenario}' in one of the files.")

            # Load and verify dimensions (we expect activations to have shape [*, 4096] and logits [*, 32000])
            acts200 = data200['activations_layer_13'][:]
            acts300 = data300['activations_layer_13'][:]
            if acts200.shape[1:] != acts300.shape[1:]:
                raise ValueError(f"Mismatch in activations dimensions in scenario '{scenario}'.")

            logits200 = data200['logits'][:]
            logits300 = data300['logits'][:]
            if logits200.shape[1:] != logits300.shape[1:]:
                raise ValueError(f"Mismatch in logits dimensions in scenario '{scenario}'.")

            # Concatenate along the sample dimension (axis 0)
            combined_acts = np.concatenate([acts200, acts300], axis=0)
            combined_logits = np.concatenate([logits200, logits300], axis=0)

            # --- Text group: prompts, matching_answers, non_matching_answers ---
            text200 = grp200['text']
            text300 = grp300['text']
            combined_text = {}
            for tkey in ['prompts', 'matching_answers', 'non_matching_answers']:
                if tkey not in text200 or tkey not in text300:
                    raise ValueError(f"Missing text dataset '{tkey}' in scenario '{scenario}' in one of the files.")
                # Decode and convert to Python lists
                texts200 = decode_string_array(text200[tkey][:])
                texts300 = decode_string_array(text300[tkey][:])
                combined_text[tkey] = texts200 + texts300

            # Save combined data for this scenario
            combined_scenarios[scenario]['data'] = {
                'activations_layer_13': combined_acts,
                'logits': combined_logits
            }
            combined_scenarios[scenario]['text'] = combined_text

        # --- Prepare new file metadata ---
        new_attrs = {
            'model_name': f200.attrs['model_name'],
            'dataset_name': f200.attrs['dataset_name'],
            'dataset_type': f200.attrs['dataset_type'],
            'creation_date': str(datetime.datetime.now()),
            'num_samples': int(f200.attrs.get('num_samples')) + int(f300.attrs.get('num_samples'))
        }
        # Use the detailed metadata from the first file (they are the same in both, aside from num_samples)
        detailed_metadata = {}
        for key in ['dataset_info', 'model_info', 'scenarios']:
            detailed_metadata[key] = f200['metadata'].attrs[key]

    # --- Write the combined data into the new HDF5 file ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as outf:
        # Write top-level attributes
        for key, value in new_attrs.items():
            outf.attrs[key] = value

        # Create metadata group and copy over detailed metadata
        meta_grp = outf.create_group('metadata')
        for key, value in detailed_metadata.items():
            meta_grp.attrs[key] = value

        # Create scenarios group and write combined data
        scen_grp = outf.create_group('scenarios')
        for scenario, content in combined_scenarios.items():
            scenario_grp = scen_grp.create_group(scenario)
            # Data subgroup
            data_grp = scenario_grp.create_group('data')
            data_grp.create_dataset('activations_layer_13', data=content['data']['activations_layer_13'], dtype='float32')
            data_grp.create_dataset('logits', data=content['data']['logits'], dtype='float32')

            # Text subgroup
            text_grp = scenario_grp.create_group('text')
            # Create a variable-length string dtype
            vlen_str = h5py.special_dtype(vlen=str)
            for tkey in ['prompts', 'matching_answers', 'non_matching_answers']:
                texts = np.array(content['text'][tkey], dtype=object)
                text_grp.create_dataset(tkey, data=texts, dtype=vlen_str)

    logging.info(f"Combined file written to: {output_path}")


def main():
    # Get the DISK_PATH from the environment variable.
    DISK_PATH = os.environ.get('DISK_PATH')
    if DISK_PATH is None:
        raise ValueError("DISK_PATH environment variable not set")

    # Define the base directories for input (200 and 300) and output (500).
    base_dir_200 = os.path.join(DISK_PATH, "anthropic_evals_activations")
    base_dir_300 = os.path.join(DISK_PATH, "anthropic_evals_activations_300")
    base_dir_500 = os.path.join(DISK_PATH, "anthropic_evals_activations_500")

    # Iterate over all datasets (each item is assumed to be a tuple (dataset_name, dataset_type))
    for dataset_name, dataset_type in all_datasets_with_type_figure_13:
        logging.info(f"Processing dataset: {dataset_name}")
        # Build file paths
        file_200 = os.path.join(
            base_dir_200,
            dataset_name,
            f"llama2_7b_chat_{dataset_name}_activations_and_logits_for_200_samples.h5"
        )
        file_300 = os.path.join(
            base_dir_300,
            dataset_name,
            f"llama2_7b_chat_{dataset_name}_activations_and_logits_for_300_samples.h5"
        )
        output_file = os.path.join(
            base_dir_500,
            dataset_name,
            f"llama2_7b_chat_{dataset_name}_activations_and_logits_for_500_samples.h5"
        )

        # Check that both input files exist; if not, log an error and stop.
        if not os.path.exists(file_200):
            raise FileNotFoundError(f"File not found: {file_200}")
        if not os.path.exists(file_300):
            raise FileNotFoundError(f"File not found: {file_300}")

        # Combine the two files into the new output file.
        combine_files(file_200, file_300, output_file)


if __name__ == "__main__":
    main()
