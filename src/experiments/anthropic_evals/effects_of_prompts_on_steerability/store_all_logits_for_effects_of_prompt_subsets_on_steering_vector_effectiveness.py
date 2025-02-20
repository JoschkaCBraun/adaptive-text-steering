"""
store_all_logits_for_effects_of_prompt_subsets_on_steering_vector_effectiveness.py

Analyzes and stores the impact of different steering vectors on token probabilities,
using different subsets of training samples for NON-PREFILLED prompt types only.
"""

import os
import random
import h5py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from steering_vectors import SteeringVector
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13

# Constants
TRAIN_SAMPLES = 200
EVAL_SAMPLES = 250
SUBSET_SIZE_K = 40  # For top, median, bottom, random subsets
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_LAYER = 13
STEERING_MULTIPLIER = 1.0
NUM_DATASETS = 36
RANDOM_SEED = 42

# We only consider these non-prefilled prompt types:
NON_PREFILLED_PROMPT_TYPES = [
    'instruction',
    '5-shot',
    'instruction_5-shot'
]

def clean_string(s: str) -> str:
    """Remove problematic characters (e.g., null bytes)."""
    return s.replace("\x00", "")

def select_subset_indices(eff_scores, k=SUBSET_SIZE_K, random_seed=RANDOM_SEED):
    """
    Given a list of effectiveness scores (length=200), return indices for:
      - full: all 200
      - top_k: top k
      - median_k: k around the median
      - bottom_k: bottom k
      - random_k: random k
    """
    # Sort by effectiveness (ascending)
    sorted_indices = sorted(range(len(eff_scores)), key=lambda i: eff_scores[i])
    
    # Bottom K (lowest effectiveness)
    bottom_k_indices = sorted_indices[:k]
    # Top K (highest effectiveness)
    top_k_indices = sorted_indices[-k:]
    
    # Median K: 40 around the median
    mid = len(eff_scores) // 2  # For 200 samples, mid=100
    median_k_start = max(0, mid - (k // 2))  # e.g., 100 - 20 = 80
    median_k_end = median_k_start + k        # 80 + 40 = 120
    median_k_indices = sorted_indices[median_k_start:median_k_end]
    
    # Random K
    rng = random.Random(random_seed)
    random_k_indices = rng.sample(range(len(eff_scores)), k)
    
    return {
        "full": list(range(len(eff_scores))),
        "top_k": top_k_indices,
        "median_k": median_k_indices,
        "bottom_k": bottom_k_indices,
        "random_k": random_k_indices,
    }

def compute_effectiveness_scores(prompts, m_answers, nm_answers, model, tokenizer, device):
    """
    Compute unsteered effectiveness scores for the 200 training samples
    using difference: (matching_prob - non_matching_prob).
    """
    from src.utils import get_token_probability
    eff_scores = []
    
    with torch.no_grad():
        for prompt, m_ans, nm_ans in zip(prompts, m_answers, nm_answers):
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model(inputs.input_ids)
            logits = outputs.logits[:, -1, :]  # shape [1, vocab_size]
            
            # Probability of the matching vs non-matching answer’s first token
            m_prob = get_token_probability(logits.cpu(), m_ans, tokenizer)
            nm_prob = get_token_probability(logits.cpu(), nm_ans, tokenizer)
            eff_scores.append(m_prob - nm_prob)
    
    return eff_scores

def compute_steering_vector_for_subset(paired_data, subset_indices, device):
    """
    Given loaded paired_data (matching_activations, non_matching_activations)
    and a list of subset indices, compute the steering vector for that subset.
    """
    m_acts = [
        torch.tensor(paired_data['matching_activations'][i], dtype=torch.float32, device=device)
        for i in subset_indices
    ]
    nm_acts = [
        torch.tensor(paired_data['non_matching_activations'][i], dtype=torch.float32, device=device)
        for i in subset_indices
    ]
    
    steering_vector = utils.compute_contrastive_steering_vector(m_acts, nm_acts, device)
    return steering_vector

def process_prompt_type(
    dataset_name: str,
    prompt_type: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    f: h5py.File
) -> None:
    """
    Process a single prompt type, storing results in the provided HDF5 file handle.
    We create the following subgroups for each prompt type:
      - full
      - top_k
      - median_k
      - bottom_k
      - random_k
    Each subgroup has:
      - steering_vector
      - prompts, matching_answers, non_matching_answers
      - full_logits, matching_logits, non_matching_logits
      - matching_probs, non_matching_probs
      - matching_answer_indices, non_matching_answer_indices
      - subset_metadata (containing effectiveness_scores, selected_indices, etc.)
    """
    from src.utils import get_token_probability, get_answer_token_ids
    
    # Create group for this prompt type
    prompt_type_group = f.create_group(prompt_type)
    
    # 1) Load the 200 training samples + their stored activations/logits
    paired_data = utils.load_paired_activations_and_logits(
        dataset_name=dataset_name,
        pairing_type=prompt_type,
        num_samples=TRAIN_SAMPLES
    )
    
    # 2) Get the actual training prompts/answers (to compute effectiveness)
    train_prompts, train_m_answers, train_nm_answers = utils.generate_prompt_and_anwers(
        dataset_name=dataset_name,
        num_samples=TRAIN_SAMPLES,
        start_index=0
    )
    
    # 3) Compute the unsteered effectiveness scores
    eff_scores = compute_effectiveness_scores(train_prompts, train_m_answers, train_nm_answers, model, tokenizer, device)
    
    # 4) Determine subsets
    subset_dict = select_subset_indices(eff_scores, k=SUBSET_SIZE_K, random_seed=RANDOM_SEED)
    
    # 5) Generate EVAL prompts for final evaluation
    eval_prompts, m_answers, nm_answers = utils.generate_prompt_and_anwers(
        dataset_name=dataset_name,
        num_samples=EVAL_SAMPLES,
        start_index=200
    )
    
    # 6) For each subset, compute a steering vector and store results
    for subset_name, subset_indices in subset_dict.items():
        subset_group = prompt_type_group.create_group(subset_name)
        
        # Store subset metadata
        subset_meta_group = subset_group.create_group("subset_metadata")
        subset_meta_group.create_dataset("effectiveness_scores", data=np.array(eff_scores, dtype=np.float32))
        subset_meta_group.create_dataset("selected_indices", data=np.array(subset_indices, dtype=np.int32))
        if subset_name == "random_k":
            subset_meta_group.attrs["random_seed"] = RANDOM_SEED
        
        # Compute the steering vector from this subset
        steering_vector = compute_steering_vector_for_subset(paired_data, subset_indices, device)
        subset_group.create_dataset('steering_vector', data=steering_vector.cpu())
        
        # Store the evaluation prompts/answers
        subset_group.create_dataset('prompts', data=[clean_string(p).encode('utf-8') for p in eval_prompts])
        subset_group.create_dataset('matching_answers', data=[clean_string(a).encode('utf-8') for a in m_answers])
        subset_group.create_dataset('non_matching_answers', data=[clean_string(a).encode('utf-8') for a in nm_answers])
        
        # Collect logits with steering
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
                # Apply the steering
                with SteeringVector({MODEL_LAYER: steering_vector}, "decoder_block").apply(
                    model, multiplier=STEERING_MULTIPLIER
                ):
                    outputs = model(inputs.input_ids)
                    logits = outputs.logits[:, -1, :].cpu()
                
                m_idx, nm_idx = get_answer_token_ids(m_answer, nm_answer, tokenizer)
                
                all_logits.append(logits)
                m_logits.append(logits[0, m_idx])
                nm_logits.append(logits[0, nm_idx])
                m_probs.append(get_token_probability(logits, m_answer, tokenizer))
                nm_probs.append(get_token_probability(logits, nm_answer, tokenizer))
                m_indices.append(m_idx)
                nm_indices.append(nm_idx)
        
        # Convert to numpy for HDF5
        all_logits_np = torch.stack(all_logits, dim=0).numpy()  # (eval_samples, 1, vocab_size)
        m_logits_np = torch.stack(m_logits, dim=0).unsqueeze(1).numpy()
        nm_logits_np = torch.stack(nm_logits, dim=0).unsqueeze(1).numpy()
        m_probs_np = np.array(m_probs, dtype=np.float32)
        nm_probs_np = np.array(nm_probs, dtype=np.float32)
        m_indices_np = np.array(m_indices, dtype=np.int32)
        nm_indices_np = np.array(nm_indices, dtype=np.int32)
        
        subset_group.create_dataset('full_logits', data=all_logits_np)
        subset_group.create_dataset('matching_logits', data=m_logits_np)
        subset_group.create_dataset('non_matching_logits', data=nm_logits_np)
        subset_group.create_dataset('matching_probs', data=m_probs_np)
        subset_group.create_dataset('non_matching_probs', data=nm_probs_np)
        subset_group.create_dataset('matching_answer_indices', data=m_indices_np)
        subset_group.create_dataset('non_matching_answer_indices', data=nm_indices_np)


def main() -> None:
    # For reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Setup device
    device = utils.get_device()
    torch.set_grad_enabled(False)
    
    # Output directory
    output_dir = os.path.join(
        utils.get_path('DISK_PATH'),
        'anthropic_evals_results',
        'effects_of_prompt_subsets_size_40_strength_1_on_steerability'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model/tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    datasets = all_datasets_with_type_figure_13[:NUM_DATASETS]
    
    with torch.no_grad():
        for dataset_name, _ in datasets:
            try:
                output_file = os.path.join(
                    output_dir,
                    f"{dataset_name}_steered_subsets_{EVAL_SAMPLES}_samples.h5"
                )
                print(f"\nProcessing dataset: {dataset_name}")
                
                with h5py.File(output_file, 'w') as f:
                    # Global metadata
                    metadata = f.create_group('metadata')
                    metadata.attrs['steering_multiplier'] = STEERING_MULTIPLIER
                    metadata.attrs['model_layer'] = MODEL_LAYER
                    metadata.attrs['train_samples'] = TRAIN_SAMPLES
                    metadata.attrs['eval_samples'] = EVAL_SAMPLES
                    metadata.attrs['model_name'] = MODEL_NAME
                    metadata.attrs['subset_size_k'] = SUBSET_SIZE_K
                    
                    # Process each non-prefilled prompt type
                    for prompt_type in NON_PREFILLED_PROMPT_TYPES:
                        process_prompt_type(dataset_name, prompt_type, model, tokenizer, device, f)
            
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
