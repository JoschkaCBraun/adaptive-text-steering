"""
plot_token_probs_for_different_prompts.py
"""
import os
import pickle
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from scipy import stats
import src.utils as utils

def get_token_variants(tokenizer: AutoTokenizer) -> Dict[str, Dict[str, List[int]]]:
    """Get token IDs for all variants of Yes/No and A/B."""
    variants = {
        "Yes": [
            tokenizer(" Yes").input_ids[-1],
            tokenizer("Yes").input_ids[-1],
            tokenizer("Yes ").input_ids[-1]
            ],
        "No": [
            tokenizer(" No").input_ids[-1],
            tokenizer("No").input_ids[-1],
            tokenizer("No ").input_ids[-1]
            ],
        "A": [
            tokenizer(" A").input_ids[-1],
            tokenizer("A").input_ids[-1],
            tokenizer("A)").input_ids[0],
            tokenizer(" A)").input_ids[0]
            ],
        "B": [
            tokenizer(" B").input_ids[-1],
            tokenizer("B").input_ids[-1],
            tokenizer("B)").input_ids[0],  # Take first token only
            tokenizer(" B)").input_ids[0]  # Take first token only
            ]}
    
    # Remove duplicates while preserving order
    for key in variants:
        variants[key] = list(dict.fromkeys(variants[key]))
    
    return variants

def get_tokens_for_answer(answer: str, token_variants: Dict[str, List[int]]) -> List[int]:
    """Map an answer to its corresponding token variants."""
    if " (A)" in answer:
        return token_variants["A"]
    elif " (B)" in answer:
        return token_variants["B"]
    elif " Yes" in answer:
        return token_variants["Yes"]
    elif " No" in answer:
        return token_variants["No"]
    else:
        raise ValueError(f"Unknown answer format: {answer}")

def compute_combined_logits(logit_distribution: torch.Tensor, token_ids: List[int]) -> Tuple[float, Dict[int, float]]:
    """
    Compute combined logits for multiple token variants using the log-sum-exp trick.
    Returns both the combined value and individual contributions.
    """
    relevant_logits = logit_distribution[token_ids]
    max_logit = torch.max(relevant_logits)
    
    # Compute exp(logit - max_logit) for numerical stability
    exp_logits = torch.exp(relevant_logits - max_logit)
    combined_value = torch.log(torch.sum(exp_logits)) + max_logit
    
    # Compute contribution of each token
    contributions = {}
    total_exp = torch.sum(exp_logits)
    for idx, token_id in enumerate(token_ids):
        contributions[token_id] = (exp_logits[idx] / total_exp).item()
    
    return combined_value.item(), contributions

def compute_confidence_metrics(logit_distribution: torch.Tensor,
                             answer_matching: str,
                             answer_not_matching: str,
                             token_variants: Dict[str, List[int]]) -> Dict:
    """Compute confidence metrics using dynamically determined tokens based on answers."""
    # Get appropriate tokens for each answer
    matching_tokens = get_tokens_for_answer(answer_matching, token_variants)
    non_matching_tokens = get_tokens_for_answer(answer_not_matching, token_variants)

    # Get combined values for each answer type
    matching_combined, matching_contributions = compute_combined_logits(logit_distribution, matching_tokens)
    non_matching_combined, non_matching_contributions = compute_combined_logits(logit_distribution, non_matching_tokens)
    
    # Convert to probabilities using softmax
    logits_tensor = torch.tensor([matching_combined, non_matching_combined])
    probs = F.softmax(logits_tensor, dim=0)
    
    return {
        'matching_prob': probs[0].item(),
        'non_matching_prob': probs[1].item(),
        'matching_contributions': matching_contributions,
        'non_matching_contributions': non_matching_contributions
    }

def plot_distributions(data: dict, dataset_name: str, token_variants: dict, 
                      ax: plt.Axes, fig: plt.Figure) -> None:
    """Plot probability distributions with enhanced visualization."""
    colors = {
        'non_matching_instruction_few_shot': '#1f77b4',
        'non_matching_instruction': '#63afd7',
        'base': '#808080',
        'matching_instruction': '#ff9999',
        'matching_instruction_few_shot': '#dc3912'
    }
    
    scenarios = [
        'non_matching_instruction_few_shot',
        'non_matching_instruction',
        'base',
        'matching_instruction',
        'matching_instruction_few_shot'
    ]
    
    x_positions = {
        'matching': {scenario: i+1 for i, scenario in enumerate(scenarios)},
        'non_matching': {scenario: i+7 for i, scenario in enumerate(scenarios)}
    }
    
    for answer_type in ['matching', 'non_matching']:
        for scenario in scenarios:
            metrics_list = []
            
            for sample in data['results'][scenario]:
                logit_distribution = sample['logit_distribution'][0].clone().detach()
                metrics = compute_confidence_metrics(
                    logit_distribution,
                    sample['answer_matching_behavior'],
                    sample['answer_not_matching_behavior'],
                    token_variants
                )
                metrics_list.append(metrics)
            
            x_pos = x_positions[answer_type][scenario]
            
            # Compute statistics
            probs = [m[f'{answer_type}_prob'] for m in metrics_list]
            mean_prob = np.mean(probs)
            sem_prob = stats.sem(probs)
            
            # Create base label
            base_label = {
                'non_matching_instruction_few_shot': 'Non-matching + Few-shot',
                'non_matching_instruction': 'Non-matching',
                'base': 'Base Model',
                'matching_instruction': 'Matching',
                'matching_instruction_few_shot': 'Matching + Few-shot'
            }[scenario] if answer_type == 'matching' else "_nolegend_"
            
            # Plot probabilities with error bars and means
            prob_label = f"{base_label}\nMean: {mean_prob:.3f}" if base_label != "_nolegend_" else "_nolegend_"
            ax.scatter([x_pos] * len(probs), probs,
                      color=colors[scenario], alpha=0.5, s=50,
                      label=prob_label)
            ax.errorbar(x_pos, mean_prob, yerr=sem_prob,
                       color=colors[scenario], capsize=5)
            ax.scatter([x_pos], [mean_prob],
                      color=colors[scenario], marker='*', s=200, zorder=3)
    
    # Customize plot
    # Add separator line
    ax.axvline(x=6, color='black', linestyle='--', alpha=0.5)
    
    # Add region labels at the bottom
    y_min = ax.get_ylim()[0]
    ax.text(3, y_min - (ax.get_ylim()[1] - y_min) * 0.1, 
            'Matching token', ha='center', va='top')
    ax.text(9, y_min - (ax.get_ylim()[1] - y_min) * 0.1, 
            'Non-matching token', ha='center', va='top')
    
    # Adjust plot limits to accommodate bottom labels
    ax.set_ylim(bottom=y_min - (ax.get_ylim()[1] - y_min) * 0.15)
    
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xticks([])
    ax.set_xlim(0.5, 11.5)
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             handletextpad=0.5, labelspacing=1.5)
    
    # Add titles and labels
    sample_size = len(data['results'][scenarios[0]])
    title_suffix = f"\n(n={sample_size} samples per condition)"
    
    ax.set_title(f"{dataset_name} - Token Probabilities\nHigher values indicate stronger preference for choice{title_suffix}")
    ax.set_ylabel('Probability (0-1 scale)')
    ax.set_ylim(0, 1)

def main() -> None:
    DATA_PATH = utils.get_path('DATA_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    PLOTS_PATH = os.path.join(DATA_PATH, "reliability_paper_data", "plots", "probs")
    os.makedirs(PLOTS_PATH, exist_ok=True)
    NUM_SAMPLES = 20
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    token_variants = get_token_variants(tokenizer)
    
    datasets = [("interest-in-science", "persona"), ("survival-instinct", "x-risk")]
    
    # Adjusted figure size for single column of plots
    fig, axs = plt.subplots(len(datasets), 1, figsize=(20, 8*len(datasets)), dpi=300)
    if len(datasets) == 1:
        axs = np.array([axs])
    
    for idx, (dataset_name, _) in enumerate(datasets):
        try:
            file_name = f"{dataset_name}/llama2_7b_chat_{dataset_name}_activations_and_logits_for_{NUM_SAMPLES}_samples.pkl"
            file_path = os.path.join(ACTIVATIONS_PATH, file_name)
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            plot_distributions(data, dataset_name, token_variants, axs[idx], fig)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Model Response Analysis: Token Probabilities\n' +
                 f'Analysis of {NUM_SAMPLES} samples per condition',
                 fontsize=16, y=1.02)
    
    output_file = os.path.join(PLOTS_PATH, f'llama2_7b_chat_probs_{NUM_SAMPLES}.pdf')
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()