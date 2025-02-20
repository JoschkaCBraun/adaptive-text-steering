"""
validate_the_stored_activations_and_logits.py

Validates HDF5 files containing model activations and logits,
creating a JSON summary of the data structure and sample content.
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_dataset_structure(group: h5py.Group) -> Dict[str, Any]:
    """Recursively get the structure of an HDF5 group."""
    structure = {}
    
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            structure[key] = get_dataset_structure(item)
        elif isinstance(item, h5py.Dataset):
            structure[key] = {
                'shape': item.shape,
                'dtype': str(item.dtype),
                'chunks': item.chunks
            }
    
    return structure

def get_sample_data(group: h5py.Group, num_samples: int = 3) -> Dict[str, Any]:
    """Get sample data from datasets in a group."""
    samples = {}
    
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            samples[key] = get_sample_data(item, num_samples)
        elif isinstance(item, h5py.Dataset):
            if len(item.shape) == 0:
                value = item[()]
                # Handle bytes objects
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                samples[key] = value
            else:
                # Ensure we don't try to get more samples than exist
                n_samples = min(num_samples, item.shape[0])
                if item.dtype == h5py.string_dtype():
                    # Handle potential bytes objects in string datasets
                    raw_data = item[:n_samples]
                    samples[key] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in raw_data]
                else:
                    # For numerical data, handle differently based on dataset name
                    sample_data = item[:n_samples]
                    if 'logits' in key:
                        # For logits, only compute statistics across all values
                        samples[key] = {
                            'min': float(np.min(sample_data)),
                            'max': float(np.max(sample_data)),
                            'mean': float(np.mean(sample_data)),
                            'std': float(np.std(sample_data))
                        }
                    else:
                        # For activations, store both raw values and statistics
                        samples[key] = {
                            #'raw_values': sample_data.tolist(),
                            'statistics': {
                                'min': float(np.min(sample_data)),
                                'max': float(np.max(sample_data)),
                                'mean': float(np.mean(sample_data)),
                                'std': float(np.std(sample_data))
                            }
                        }
    
    return samples

def validate_h5_file(file_path: str, output_dir: str) -> None:
    """Validate an HDF5 file and create a JSON summary."""
    logging.info(f"Validating file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Get metadata
            metadata = {
                'model_name': f.attrs['model_name'],
                'dataset_name': f.attrs['dataset_name'],
                'dataset_type': f.attrs['dataset_type'],
                'creation_date': f.attrs['creation_date'],
                'num_samples': int(f.attrs['num_samples'])
            }
            
            # Get detailed metadata from metadata group
            detailed_metadata = {
                'dataset_info': json.loads(f['metadata'].attrs['dataset_info']),
                'model_info': json.loads(f['metadata'].attrs['model_info']),
                'scenarios': json.loads(f['metadata'].attrs['scenarios'])
            }
            
            # Get data structure
            structure = get_dataset_structure(f)
            
            # Get sample data from scenarios
            scenario_samples = get_sample_data(f['scenarios'])
            
            # Compile validation summary
            validation_summary = {
                'file_path': file_path,
                'validation_date': str(datetime.now()),
                'metadata': metadata,
                'detailed_metadata': detailed_metadata,
                'data_structure': structure,
                'sample_data': scenario_samples
            }
            
            # Create output filename based on input file
            base_name = os.path.basename(file_path)
            output_name = f"validation_{base_name.replace('.h5', '.json')}"
            output_path = os.path.join(output_dir, output_name)
            
            # Save validation summary
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w') as f_out:
                json.dump(validation_summary, f_out, indent=2)
            
            logging.info(f"Validation summary saved to: {output_path}")
            
    except Exception as e:
        logging.error(f"Error validating file {file_path}: {str(e)}", exc_info=True)

def main() -> None:
    # Configure paths
    DISK_PATH = os.environ.get('DISK_PATH')
    DATA_PATH = os.environ.get('DATA_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "anthropic_evals_activations")
    VALIDATION_OUTPUT_PATH = os.path.join(DATA_PATH, "validation_results")
    
    # Walk through the activations directory
    for root, _, files in os.walk(ACTIVATIONS_PATH):
        for file in files:
            if file.endswith('.h5'):
                h5_path = os.path.join(root, file)
                validate_h5_file(h5_path, VALIDATION_OUTPUT_PATH)

if __name__ == "__main__":
    main()