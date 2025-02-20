"""
load_stored_anthropic_evals_activations_and_logits.py

Functions to load and validate stored activations and logits from HDF5 files,
and prepare paired activations for steering vector computation.
"""

import os
import json
from typing import Dict, List, Optional

import h5py
import numpy as np

from src.utils.dataset_names import all_datasets_figure_13

def load_paired_activations_and_logits(
    dataset_name: str,
    pairing_type: str,
    num_samples: Optional[int] = 200,
) -> Dict:
    """
    Loads paired activations and logits (matching and non-matching) for analysis and creating steering vectors.
    
    Args:
        dataset_name: Name of the dataset to load
        num_samples: Number of samples to load (max 500)
        pairing_type: Type of pairing (e.g., "prefilled_answer", "instruction", etc.)
    
    Returns:
        Dict with structure:
        {
            'matching_activations': np.ndarray,     # shape: [num_samples, 4096]
            'non_matching_activations': np.ndarray,  # shape: [num_samples, 4096]
            'matching_logits': np.ndarray,          # shape: [num_samples, 32000]
            'non_matching_logits': np.ndarray,      # shape: [num_samples, 32000]
            'metadata': {
                'pairing_type': str,
                'matching_scenario': str,
                'non_matching_scenario': str,
                'dataset_name': str,
                'num_samples': int,
                'matching_answers': List[str],    # List of matching answers
                'non_matching_answers': List[str], # List of non-matching answers
                'prompts': List[str]              # List of input prompts
            }
        }
    """
    # Load steering vector pairings
    DATASETS_PATH = os.environ.get('DATASETS_PATH')
    if DATASETS_PATH is None:
        raise ValueError("DATASETS_PATH environment variable not set")
        
    pairings_path = os.path.join(
        DATASETS_PATH,
        "anthropic_evals/prompts/steering_vector_pairings.json"
    )
    
    try:
        with open(pairings_path, 'r') as f:
            pairings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Steering vector pairings file not found at {pairings_path}")
    
    # Validate pairing type
    if pairing_type not in pairings:
        raise ValueError(f"Invalid pairing type. Must be one of: {list(pairings.keys())}")
    
    # Get scenario names for the pairing
    matching_scenario = pairings[pairing_type]['matching']
    non_matching_scenario = pairings[pairing_type]['non_matching']
    
    # Load data for both scenarios with a single call
    data = load_anthropic_evals_activations_and_logits(
        num_samples=num_samples,
        dataset_name=dataset_name,
        prompt_configurations=[matching_scenario, non_matching_scenario]
    )
    
    # Extract both activations and logits
    matching_activations = data['scenarios'][matching_scenario]['data']['activations_layer_13']
    non_matching_activations = data['scenarios'][non_matching_scenario]['data']['activations_layer_13']
    matching_logits = data['scenarios'][matching_scenario]['data']['logits']
    non_matching_logits = data['scenarios'][non_matching_scenario]['data']['logits']
    
    # Extract text data
    matching_answers = data['scenarios'][matching_scenario]['text']['matching_answers']
    non_matching_answers = data['scenarios'][matching_scenario]['text']['non_matching_answers']
    prompts = data['scenarios'][matching_scenario]['text']['prompts']
    
    return {
        'matching_activations': matching_activations,
        'non_matching_activations': non_matching_activations,
        'matching_logits': matching_logits,
        'non_matching_logits': non_matching_logits,
        'metadata': {
            'pairing_type': pairing_type,
            'matching_scenario': matching_scenario,
            'non_matching_scenario': non_matching_scenario,
            'dataset_name': dataset_name,
            'num_samples': num_samples,
            'matching_answers': matching_answers,
            'non_matching_answers': non_matching_answers,
            'prompts': prompts
        }
    }

def load_anthropic_evals_activations_and_logits(
    dataset_name: str,
    num_samples: Optional[int] = 200,
    prompt_configurations: Optional[List[str]] = None,
) -> Dict:
    """
    Load stored activations and logits for a specific dataset and configurations.
    
    Args:
        dataset_name: Name of the dataset to load
        num_samples: Number of samples to load (max 500)
        prompt_configurations: List of prompt configurations to load. If None, load all.
    
    Returns:
        Dictionary containing the loaded data with original HDF5 structure
    
    Raises:
        ValueError: If inputs are invalid or data validation fails
        FileNotFoundError: If required files don't exist
    """
    # Validate num_samples
    if not isinstance(num_samples, int) or num_samples <= 0 or num_samples > 500:
        raise ValueError(f"num_samples must be between 1 and 500, got {num_samples}")
        
    # Validate dataset name
    valid_datasets = all_datasets_figure_13
    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name. Must be one of: {valid_datasets}")
    
    # Load prompt configurations
    DATASETS_PATH = os.environ.get('DATASETS_PATH')
    if DATASETS_PATH is None:
        raise ValueError("DATASETS_PATH environment variable not set")
        
    config_path = os.path.join(
        DATASETS_PATH,
        "anthropic_evals/prompts/prompt_configurations.json"
    )
    
    try:
        with open(config_path, 'r') as f:
            available_configs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt configurations file not found at {config_path}")
        
    # Validate prompt configurations
    if prompt_configurations is not None:
        invalid_configs = [cfg for cfg in prompt_configurations if cfg not in available_configs]
        if invalid_configs:
            raise ValueError(f"Invalid prompt configurations: {invalid_configs}")
    
    # Locate and load HDF5 file
    DISK_PATH = os.environ.get('DISK_PATH')
    if DISK_PATH is None:
        raise ValueError("DISK_PATH environment variable not set")
        
    h5_path = os.path.join(
        DISK_PATH,
        "anthropic_evals_activations",
        dataset_name,
        f"llama2_7b_chat_{dataset_name}_activations_and_logits_for_{num_samples}_samples.h5"
    )
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at {h5_path}")
    
    # Load and validate data
    with h5py.File(h5_path, 'r') as f:
        # First validate file structure
        if 'scenarios' not in f:
            raise ValueError("Invalid HDF5 file: 'scenarios' group not found")
            
        # Validate requested configurations exist in file
        file_configs = list(f['scenarios'].keys())
        if prompt_configurations is not None:
            missing_configs = [cfg for cfg in prompt_configurations if cfg not in file_configs]
            if missing_configs:
                raise ValueError(f"Configurations not found in HDF5 file: {missing_configs}")
        else:
            prompt_configurations = file_configs
            
        # Load data with validation
        data = {}
        
        # Load metadata
        data['metadata'] = {
            'model_name': f.attrs['model_name'],
            'dataset_name': f.attrs['dataset_name'],
            'dataset_type': f.attrs['dataset_type'],
            'creation_date': f.attrs['creation_date'],
            'num_samples': int(f.attrs['num_samples'])
        }
        
        # Load detailed metadata
        data['detailed_metadata'] = {
            'dataset_info': json.loads(f['metadata'].attrs['dataset_info']),
            'model_info': json.loads(f['metadata'].attrs['model_info']),
            'scenarios': json.loads(f['metadata'].attrs['scenarios'])
        }
        
        # Load scenarios
        data['scenarios'] = {}
        for config in prompt_configurations:
            scenario_group = f['scenarios'][config]
            
            # Validate data structure
            required_groups = {'data', 'text'}
            if not all(group in scenario_group for group in required_groups):
                raise ValueError(f"Invalid structure in scenario {config}")
                
            # Validate dimensions
            activations = scenario_group['data/activations_layer_13']
            logits = scenario_group['data/logits']
            
            if activations.shape[1] != 4096:
                raise ValueError(f"Invalid activation dimensions in {config}: {activations.shape}")
            if logits.shape[1] != 32000:
                raise ValueError(f"Invalid logit dimensions in {config}: {logits.shape}")
                
            # Load scenario data
            data['scenarios'][config] = {
                'data': {
                    'activations_layer_13': activations[:].astype(np.float32),
                    'logits': logits[:].astype(np.float32)
                },
                'text': {
                    'prompts': [p.decode('utf-8') if isinstance(p, bytes) else p 
                            for p in scenario_group['text/prompts'][:].tolist()],
                    'matching_answers': [a.decode('utf-8') if isinstance(a, bytes) else a 
                                    for a in scenario_group['text/matching_answers'][:].tolist()],
                    'non_matching_answers': [a.decode('utf-8') if isinstance(a, bytes) else a 
                                        for a in scenario_group['text/non_matching_answers'][:].tolist()]
                }
            }
                    
    return data