'''
load_stored_anthropic_evals_activations_and_logits.py
'''
import os
import json
import logging
from typing import Dict, List, Optional
import h5py

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_prompt_configurations() -> Dict:
    """Load prompt configurations from JSON file."""
    config_path = os.path.join(utils.get_path('DATASETS_PATH'), 
                             'anthropic_evals', 'prompts', 
                             'prompt_configurations.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_h5_data_structure(f: h5py.File, scenario: str, n_samples: int) -> None:
    """Validate H5 file structure and data completeness with improved checks.
    
    Args:
        f: Open H5 file object
        scenario: Name of the scenario being validated
        n_samples: Expected number of samples
        
    Raises:
        ValueError: If validation fails
    """
    # Validate basic scenario structure
    if 'text' not in f['scenarios'][scenario]:
        raise ValueError(f"Missing 'text' group in scenario {scenario}")
    if 'data' not in f['scenarios'][scenario]:
        raise ValueError(f"Missing 'data' group in scenario {scenario}")
        
    # Get group references
    text_group = f['scenarios'][scenario]['text']
    data_group = f['scenarios'][scenario]['data']
    
    # Validate text data existence and completeness
    required_text_keys = {'prompts', 'matching_answers', 'non_matching_answers'}
    missing_keys = required_text_keys - set(text_group.keys())
    if missing_keys:
        raise ValueError(f"Missing text keys in {scenario}: {missing_keys}")
    
    # Validate sample counts in text data
    for key in required_text_keys:
        data = text_group[key]
        if len(data) != n_samples:
            raise ValueError(f"Incorrect number of samples in {scenario}/{key}. "
                           f"Expected {n_samples}, got {len(data)}")
    
    # Determine data structure type and validate accordingly
    has_regular_data = 'logits' in data_group and len(data_group['logits']) > 0
    has_prefilled = ('prefilled' in data_group and 
                    all(condition in data_group['prefilled'] 
                        for condition in ['matching', 'nonmatching']))
    
    if has_regular_data:
        # Validate regular logits and activations
        logits = data_group['logits']
        if len(logits) != n_samples:
            raise ValueError(f"Incorrect number of logits in {scenario}. "
                           f"Expected {n_samples}, got {len(logits)}")
            
        # Check for activations
        activation_key = 'activations_layer_13'
        if activation_key not in data_group:
            raise ValueError(f"Missing {activation_key} in {scenario}")
            
        activations = data_group[activation_key]
        if len(activations) != n_samples:
            raise ValueError(f"Incorrect number of activations in {scenario}. "
                           f"Expected {n_samples}, got {len(activations)}")
            
    elif has_prefilled:
        # Validate prefilled structure
        for condition in ['matching', 'nonmatching']:
            condition_group = data_group['prefilled'][condition]
            
            # Check logits
            if 'logits' not in condition_group:
                raise ValueError(f"Missing logits in {scenario}/prefilled/{condition}")
                
            logits = condition_group['logits']
            if len(logits) != n_samples:
                raise ValueError(f"Incorrect number of logits in {scenario}/prefilled/{condition}. "
                               f"Expected {n_samples}, got {len(logits)}")
                
            # Check activations
            activation_key = 'activations_layer_13'
            if activation_key not in condition_group:
                raise ValueError(f"Missing {activation_key} in {scenario}/prefilled/{condition}")
                
            activations = condition_group[activation_key]
            if len(activations) != n_samples:
                raise ValueError(f"Incorrect number of activations in {scenario}/prefilled/{condition}. "
                               f"Expected {n_samples}, got {len(activations)}")
    else:
        raise ValueError(f"No valid logits or activations data found in scenario {scenario}")

def load_stored_anthropic_evals_activations_and_logits_h5(
    dataset_name: str,
    prompt_configurations: Optional[List[str]] = None,
    num_samples: Optional[int] = None
) -> Dict:
    """
    Load stored activations and logits from H5 files with comprehensive validation.
    
    Args:
        dataset_name: Name of the dataset to load
        prompt_configurations: Optional list of configurations to load
        num_samples: Number of samples to load (max 200)
    
    Returns:
        Dict with validated data structure containing:
            - metadata: Dataset and model information
            - scenarios: Dictionary of scenarios, each containing:
                - text: prompts and answers
                - data: logits and activations
    
    Raises:
        ValueError: If data validation fails
        FileNotFoundError: If H5 file not found
    """
    MAX_SAMPLES = 200
    MODEL_NAME = 'llama2_7b_chat'

    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Requested samples: {num_samples if num_samples else MAX_SAMPLES}")
    
    # Validate dataset name
    valid_datasets = [name for name, _ in all_datasets_with_type_figure_13_tan_et_al]
    if dataset_name not in valid_datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found. Must be one of: {valid_datasets}")
    
    # Validate number of samples
    if num_samples is not None:
        if not isinstance(num_samples, int) or num_samples <= 0 or num_samples > MAX_SAMPLES:
            raise ValueError(f"num_samples must be a positive integer <= {MAX_SAMPLES}")
    n_samples = num_samples if num_samples is not None else MAX_SAMPLES
    
    # Load and validate prompt configurations
    valid_configs = load_prompt_configurations()
    if prompt_configurations is not None:
        invalid_configs = set(prompt_configurations) - set(valid_configs.keys())
        if invalid_configs:
            raise ValueError(f"Invalid prompt configurations: {invalid_configs}. "
                           f"Must be among: {list(valid_configs.keys())}")
    
    # Construct and validate file path
    file_path = os.path.join(
        utils.get_path('DISK_PATH'),
        'anthropic_evals_activations',
        dataset_name,
        f"{MODEL_NAME}_{dataset_name}_activations_and_logits_for_{MAX_SAMPLES}_samples.h5"
    )
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"H5 file not found at: {file_path}")
    
    # Initialize result dictionary
    result = {}
    
    with h5py.File(file_path, 'r') as f:
        logger.info(f"Validating file structure for dataset: {dataset_name}")
        
        # Validate basic file structure
        if 'metadata' not in f:
            raise ValueError("Missing 'metadata' group in H5 file")
        if 'scenarios' not in f:
            raise ValueError("Missing 'scenarios' group in H5 file")
            
        # Load and validate metadata
        required_attrs = {'dataset_info', 'model_info', 'scenarios'}
        missing_attrs = required_attrs - set(f['metadata'].attrs.keys())
        if missing_attrs:
            raise ValueError(f"Missing metadata attributes: {missing_attrs}")
            
        # Load metadata
        result['metadata'] = {
            'dataset_info': json.loads(f['metadata'].attrs['dataset_info']),
            'model_info': json.loads(f['metadata'].attrs['model_info']),
            'scenarios': json.loads(f['metadata'].attrs['scenarios'])
        }
        
        # Get available scenarios
        available_scenarios = list(f['scenarios'].keys())
        logger.info(f"Available scenarios: {available_scenarios}")
        
        # Determine which scenarios to load
        scenarios_to_load = (prompt_configurations if prompt_configurations is not None 
                           else available_scenarios)
        
        # Initialize scenarios in result
        result['scenarios'] = {}
        
        # Load and validate each scenario
        for scenario in scenarios_to_load:
            if scenario not in available_scenarios:
                logger.warning(f"Scenario '{scenario}' not found in file, skipping")
                continue
                            
            try:
                # Validate scenario structure
                validate_h5_data_structure(f, scenario, n_samples)
                
                # Initialize scenario structure
                result['scenarios'][scenario] = {
                    'text': {},
                    'data': {}
                }
                
                # Load text data
                for text_key in ['prompts', 'matching_answers', 'non_matching_answers']:
                    data = f[f'scenarios/{scenario}/text/{text_key}'][:n_samples]
                    result['scenarios'][scenario]['text'][text_key] = [
                        s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in data
                    ]
                
                # Load numerical data based on structure
                data_group = f[f'scenarios/{scenario}/data']
                
                if 'logits' in data_group and len(data_group['logits']) > 0:
                    # Load regular data
                    result['scenarios'][scenario]['data'].update({
                        'logits': data_group['logits'][:n_samples],
                        'activations_layer_13': data_group['activations_layer_13'][:n_samples]
                    })
                else:
                    # Load prefilled data
                    for condition in ['matching', 'nonmatching']:
                        prefix = f"prefilled_{condition}"
                        prefilled_group = data_group['prefilled'][condition]
                        result['scenarios'][scenario]['data'][prefix] = {
                            'logits': prefilled_group['logits'][:n_samples],
                            'activations_layer_13': prefilled_group['activations_layer_13'][:n_samples]
                        }
                        
            except Exception as e:
                logger.error(f"Error loading scenario {scenario}: {str(e)}")
                continue
    
    if not result['scenarios']:
        raise ValueError("No valid scenarios were loaded")
    
    logger.info("Data loading complete!")
    return result

def print_scenario_info(scenario_name: str, scenario_data: dict):
    """Helper function to print scenario information in a structured way."""
    print(f"\nScenario: {scenario_name}")
    print("-" * 50)
    
    # Print text data
    print("Sample prompts (first 2):")
    for i, prompt in enumerate(scenario_data['text']['prompts'][:2]):
        print(f"\nPrompt {i+1}:")
        print(prompt)
    
    print("\nSample matching answers (first 2):")
    for i, answer in enumerate(scenario_data['text']['matching_answers'][:2]):
        print(f"{i+1}. {answer}")
    
    print("\nSample non-matching answers (first 2):")
    for i, answer in enumerate(scenario_data['text']['non_matching_answers'][:2]):
        print(f"{i+1}. {answer}")
    
    # Print data shape information
    print("\nData shapes:")
    if 'prefilled_matching' in scenario_data['data']:
        print("Prefilled matching logits shape:", 
              scenario_data['data']['prefilled_matching']['logits'].shape)
        print("Prefilled non-matching logits shape:", 
              scenario_data['data']['prefilled_nonmatching']['logits'].shape)
    else:
        print("Logits shape:", scenario_data['data']['logits'].shape)

def main():
    # Datasets to analyze
    datasets = ['corrigible-neutral-HHH', 'interest-in-science']
    
    # Load all available prompt configurations
    prompt_configs = load_prompt_configurations()
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")
        
        try:
            # Load data for the dataset (using default 200 samples)
            results = load_stored_anthropic_evals_activations_and_logits_h5(dataset, num_samples=200)
            
            # Print metadata
            print("\nMetadata:")
            print("-" * 50)
            print("Dataset info:", results['metadata']['dataset_info'])
            print("Model info:", results['metadata']['model_info'])
            
            # Print information for each scenario
            for scenario_name, scenario_data in results['scenarios'].items():
                print_scenario_info(scenario_name, scenario_data)
                
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            continue

if __name__ == "__main__":
    main()