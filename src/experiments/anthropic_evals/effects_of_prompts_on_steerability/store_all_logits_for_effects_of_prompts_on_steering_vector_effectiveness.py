"""
store_all_logits_for_effects_of_prompts_on_steering_vector_effectiveness.py
Analyzes and stores the impact of different steering vectors on token probabilities.
"""
import os
import h5py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import SteeringVector

import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13

# Constants
TRAIN_SAMPLES = 250
EVAL_SAMPLES = 250
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_LAYER = 13
STEERING_MULTIPLIER = 1.0
NUM_DATASETS = 36
PROMPT_TYPES = [
    'prefilled_answer',
    'instruction',
    '5-shot',
    'prefilled_instruction',
    'prefilled_5-shot',
    'instruction_5-shot',
    'prefilled_instruction_5-shot'
]

def process_dataset(dataset_name: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                    device: str, output_file: str) -> None:
    """Process a single dataset, computing and storing logits for all prompt types including baseline."""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Generate evaluation prompts
    eval_prompts, m_answers, nm_answers = utils.generate_prompt_and_anwers(
        dataset_name=dataset_name,
        num_samples=EVAL_SAMPLES,
        start_index=200)
    
    # Create h5 file for this dataset
    with h5py.File(output_file, 'w') as f:
        # Store metadata
        metadata = f.create_group('metadata')
        metadata.attrs['steering_multiplier'] = STEERING_MULTIPLIER
        metadata.attrs['model_layer'] = MODEL_LAYER
        metadata.attrs['train_samples'] = TRAIN_SAMPLES
        metadata.attrs['eval_samples'] = EVAL_SAMPLES
        metadata.attrs['model_name'] = MODEL_NAME
        
        # Process all prompt types (including baseline as "no_steering")
        all_prompt_types = ['no_steering'] + PROMPT_TYPES
        
        for prompt_type in all_prompt_types:
            try:
                prompt_group = f.create_group(prompt_type)
                
                # For baseline case, create zero steering vector
                if prompt_type == 'no_steering':
                    # Option 1: Zero tensor
                    steering_vector = torch.zeros(model.config.hidden_size, device=device)
                    # Option 2: Set multiplier to 0 instead
                    current_multiplier = 0.0
                else:
                    # Load paired activations for steering vector computation
                    paired_data = utils.load_paired_activations_and_logits(
                        dataset_name=dataset_name,
                        pairing_type=prompt_type,
                        num_samples=TRAIN_SAMPLES
                    )
                    
                    # Compute steering vector
                    m_tensors = [torch.tensor(row, dtype=torch.float32).to(device) 
                                for row in paired_data['matching_activations']]
                    nm_tensors = [torch.tensor(row, dtype=torch.float32).to(device) 
                                for row in paired_data['non_matching_activations']]
                    
                    steering_vector = utils.compute_contrastive_steering_vector(m_tensors, nm_tensors, device)
                    current_multiplier = STEERING_MULTIPLIER
                
                # Store steering vector (zeros for baseline)
                prompt_group.create_dataset('steering_vector', data=steering_vector.cpu())
                
                # Store prompts and answers
                prompt_group.create_dataset('prompts', data=[clean_string(p).encode('utf-8') for p in eval_prompts])
                prompt_group.create_dataset('matching_answers', data=[clean_string(a).encode('utf-8') for a in m_answers])
                prompt_group.create_dataset('non_matching_answers', data=[clean_string(a).encode('utf-8') for a in nm_answers])

                
                # Get logits with steering (or zero steering for baseline)
                all_logits = []
                m_logits = []
                nm_logits = []
                m_probs = []
                nm_probs = []
                m_indices = []
                nm_indices = []
                
                with torch.no_grad():
                    for prompt, m_answer, nm_answer in zip(eval_prompts, m_answers, nm_answers):
                        inputs = tokenizer(prompt, return_tensors='pt').to(device)
                        
                        # Apply steering vector (zero vector/multiplier for baseline)
                        with SteeringVector({MODEL_LAYER: steering_vector}, "decoder_block").apply(
                            model, multiplier=current_multiplier):
                            outputs = model(inputs.input_ids)
                            logits = outputs.logits[:, -1, :].cpu()
                        
                        m_idx, nm_idx = utils.get_answer_token_ids(m_answer, nm_answer, tokenizer)
                        
                        all_logits.append(logits)
                        m_logits.append(logits[0, m_idx])
                        nm_logits.append(logits[0, nm_idx])
                        m_probs.append(utils.get_token_probability(logits, m_answer, tokenizer))
                        nm_probs.append(utils.get_token_probability(logits, nm_answer, tokenizer))
                        m_indices.append(m_idx)
                        nm_indices.append(nm_idx)
                
                # Store results
                prompt_group.create_dataset('full_logits', data=all_logits)
                prompt_group.create_dataset('matching_logits', data=m_logits)
                prompt_group.create_dataset('non_matching_logits', data=nm_logits)
                prompt_group.create_dataset('matching_probs', data=m_probs)
                prompt_group.create_dataset('non_matching_probs', data=nm_probs)
                prompt_group.create_dataset('matching_answer_indices', data=m_indices)
                prompt_group.create_dataset('non_matching_answer_indices', data=nm_indices)
                
            except Exception as e:
                print(f"Error processing prompt type {prompt_type}: {str(e)}")
                continue

def clean_string(s):
    return s.replace("\x00", "")


def main() -> None:
    # Setup
    device = utils.get_device()
    torch.set_grad_enabled(False)
    
    output_dir = os.path.join(utils.get_path('DISK_PATH'), 'anthropic_evals_results', 'anti_effects_of_prompts_on_steerability')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # Process datasets
    datasets = all_datasets_with_type_figure_13[:NUM_DATASETS]
    
    with torch.no_grad():
        for dataset_name, _ in datasets:
            try:
                output_file = os.path.join(
                    output_dir,
                    f"{dataset_name}_anti_steered_and_baseline_logits_{EVAL_SAMPLES}_samples_{NUM_DATASETS}_datasets.h5"
                )
                process_dataset(dataset_name, model, tokenizer, device, output_file)
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue

if __name__ == "__main__":
    main()