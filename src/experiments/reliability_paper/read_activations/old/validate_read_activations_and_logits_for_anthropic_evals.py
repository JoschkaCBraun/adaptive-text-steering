"""
validate_h5_structure.py
Utility script to validate and inspect HDF5 files containing activation data.
"""

import h5py
import json
import logging
import numpy as np
from typing import Any, Dict
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_attrs(name: str, obj: Any) -> None:
    """Print all attributes of an HDF5 object."""
    if len(obj.attrs) > 0:
        print(f"\nAttributes for {name}:")
        for key, value in obj.attrs.items():
            # Handle JSON-encoded attributes
            if key in ['dataset_info', 'model_info', 'scenarios']:
                value = json.loads(value)
            print(f"  {key}: {value}")

def print_dataset_info(name: str, dataset: h5py.Dataset) -> None:
    """Print detailed information about a dataset."""
    print(f"\nDataset: {name}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Dtype: {dataset.dtype}")
    print(f"  Chunks: {dataset.chunks}")
    print(f"  Compression: {dataset.compression}")
    
    # Print sample data
    if dataset.size > 0:
        if dataset.dtype == h5py.string_dtype():
            sample = dataset[0]
            print(f"  First element (string): {sample}")
        else:
            sample = dataset[0]
            print(f"  First element shape: {sample.shape}")
            print(f"  Value range: [{np.min(sample):.3f}, {np.max(sample):.3f}]")
            print(f"  Mean: {np.mean(sample):.3f}")

def validate_h5_file(file_path: str) -> None:
    """Validate and print detailed information about the HDF5 file structure."""
    with h5py.File(file_path, 'r') as f:
        print(f"\n{'='*80}\nValidating file: {file_path}\n{'='*80}")
        
        # Print top-level attributes
        print_attrs("Root", f)
        
        def visit_and_validate(name: str, obj: Any) -> None:
            """Recursively visit and validate all groups and datasets."""
            if isinstance(obj, h5py.Dataset):
                print_dataset_info(name, obj)
                print_attrs(name, obj)
            elif isinstance(obj, h5py.Group):
                print(f"\nGroup: {name}/")
                print_attrs(name, obj)
        
        # Visit all groups and datasets
        f.visititems(visit_and_validate)
        
        # Additional validation for specific structure
        for scenario_name in f['scenarios'].keys():
            scenario_path = f'scenarios/{scenario_name}'
            
            # Validate text data exists
            text_group = f[f'{scenario_path}/text']
            required_text = ['prompts', 'matching_answers', 'non_matching_answers']
            for req in required_text:
                assert req in text_group, f"Missing {req} in {scenario_path}/text"
            
            # Validate data structure
            data_group = f[f'{scenario_path}/data']
            if 'prefilled' in data_group:
                # Validate prefilled structure
                for condition in ['matching', 'nonmatching']:
                    condition_path = f'{scenario_path}/data/prefilled/{condition}'
                    assert 'logits' in f[condition_path], f"Missing logits in {condition_path}"
                    
                    # Check activation layers
                    acts = [k for k in f[condition_path].keys() if k.startswith('activations_layer_')]
                    print(f"\nFound activation layers in {condition_path}: {acts}")
            else:
                # Validate non-prefilled structure
                assert 'logits' in data_group, f"Missing logits in {scenario_path}/data"
                acts = [k for k in data_group.keys() if k.startswith('activations_layer_')]
                print(f"\nFound activation layers in {scenario_path}/data: {acts}")

def main():
    # Example usage
    DISK_PATH = os.environ.get('DISK_PATH', './data')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "anthropic_evals_activations")
    
    # Find all .h5 files recursively
    for root, _, files in os.walk(ACTIVATIONS_PATH):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                try:
                    validate_h5_file(file_path)
                except Exception as e:
                    logging.error(f"Error validating {file_path}: {e}", exc_info=True)

if __name__ == "__main__":
    main()