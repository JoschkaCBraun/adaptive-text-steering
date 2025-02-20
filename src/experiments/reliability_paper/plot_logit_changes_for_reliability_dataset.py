import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils
from src.utils.dataset_names import all_datasets_with_type_figure_13_tan_et_al

def load_and_validate_data(file_path: str) -> dict:
    """Load data and validate logits and probabilities."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        if not isinstance(data, dict):
            data = {'samples': data}
        elif 'samples' not in data and any(isinstance(item, dict) for item in data):
            data = {'samples': list(data.values())}
    
    # Validate data
    for sample in data['samples']:
        for scenario in ['base', 'negative_prompt', 'positive_prompt']:
            scenario_key = 'base' if scenario == 'base' else scenario
            try:
                for token_type in ['positive', 'negative']:
                    token_name = sample[f'{token_type}_token']
                    logit = sample['scenarios'][scenario_key]['logits'][token_type][token_name]['logit']
                    prob = sample['scenarios'][scenario_key]['logits'][token_type][token_name]['prob']
                    
                    if not isinstance(logit, (int, float)):
                        raise ValueError(f"Invalid logit value for {scenario_key}, {token_type}: {logit}")
                    if not isinstance(prob, (int, float)) or not 0 <= prob <= 1:
                        raise ValueError(f"Invalid probability value for {scenario_key}, {token_type}: {prob}")
            except KeyError as e:
                print(f"Missing data in sample: {e}")
                continue
    return data

def plot_distributions(data: dict, dataset_name: str, ax_logits, ax_probs) -> None:
    """Plot logit and probability distributions."""
    colors = {'negative_prompt': 'blue', 'base': 'gray', 'positive_prompt': 'red'}
    scenarios = ['negative_prompt', 'base', 'positive_prompt']
    x_positions = {
        'positive': {scenario: i+1 for i, scenario in enumerate(scenarios)},
        'negative': {scenario: i+4 for i, scenario in enumerate(scenarios)}
    }
    
    samples = data['samples']
    if not samples:
        for ax in [ax_logits, ax_probs]:
            ax.text(0.5, 0.5, f'No data available for {dataset_name}', 
                    ha='center', va='center')
        return
    
    pos_token = samples[0]['positive_token']
    neg_token = samples[0]['negative_token']
    
    for token_type, token_name in [('positive', pos_token), ('negative', neg_token)]:
        for scenario in scenarios:
            scenario_key = 'base' if scenario == 'base' else scenario
            try:
                logits = []
                probs = []
                for sample in samples:
                    data_dict = sample['scenarios'][scenario_key]['logits'][token_type][token_name]
                    logits.append(data_dict['logit'])
                    probs.append(data_dict['prob'])
                
                x_pos = x_positions[token_type][scenario]
                
                # Plot logits
                ax_logits.scatter([x_pos] * len(logits), logits, color=colors[scenario], 
                                alpha=0.5, s=50)
                ax_logits.scatter([x_pos], [np.mean(logits)], color=colors[scenario], 
                                marker='*', s=200, zorder=3)
                
                # Plot probabilities
                ax_probs.scatter([x_pos] * len(probs), probs, color=colors[scenario], 
                               alpha=0.5, s=50)
                ax_probs.scatter([x_pos], [np.mean(probs)], color=colors[scenario], 
                               marker='*', s=200, zorder=3)
                
            except KeyError as e:
                print(f"Warning: Missing data for {dataset_name}, {scenario}: {e}")
                continue
    
    for ax in [ax_logits, ax_probs]:
        ax.axvline(x=3.5, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticks([])
        ax.set_xlim(0.5, 6.5)
    
    ax_logits.set_title(f"{dataset_name} - Logits")
    ax_probs.set_title(f"{dataset_name} - Probabilities")
    ax_logits.set_ylabel('Logit Value')
    ax_probs.set_ylabel('Probability')
    ax_probs.set_ylim(0, 1)

def main() -> None:
    DATA_PATH = utils.get_path('DATA_PATH')
    DISK_PATH = utils.get_path('DISK_PATH')
    ACTIVATIONS_PATH = os.path.join(DISK_PATH, "reliability_paper_activations")
    PLOTS_PATH = os.path.join(DATA_PATH, "reliability_paper_data", "plots", "combined")
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    TARGET_SAMPLES = 25
    datasets = [name for name, _ in all_datasets_with_type_figure_13_tan_et_al]
    models = [('llama2_7b_chat', 'meta-llama/Llama-2-7b-chat-hf', [13])]
    
    for model_name, _, _ in models:
        fig, axs = plt.subplots(len(datasets), 2, figsize=(20, 6*len(datasets)))
        if len(datasets) == 1:
            axs = axs.reshape(1, -1)
        
        for dataset_idx, dataset_name in enumerate(datasets):
            dataset_dir = os.path.join(ACTIVATIONS_PATH, dataset_name)
            if not os.path.exists(dataset_dir):
                print(f"Warning: Directory not found for dataset {dataset_name}")
                continue
                
            try:
                target_file = next((f for f in os.listdir(dataset_dir) 
                                  if f.endswith("_with_logits.pkl") 
                                  and model_name in f 
                                  and f"for_{TARGET_SAMPLES}_samples" in f), None)
                
                if target_file:
                    file_path = os.path.join(dataset_dir, target_file)
                    data = load_and_validate_data(file_path)
                    plot_distributions(data, dataset_name, axs[dataset_idx, 0], axs[dataset_idx, 1])
                else:
                    print(f"Warning: No matching file found for {dataset_name}")
                    
            except (FileNotFoundError, StopIteration) as e:
                print(f"Error processing dataset {dataset_name}: {e}")
                continue
            except ValueError as e:
                print(f"Data validation error for {dataset_name}: {e}")
                continue
        
        plt.tight_layout()
        output_file = os.path.join(PLOTS_PATH, 
                                 f'{model_name}_logits_and_probs_{TARGET_SAMPLES}_samples.pdf')
        fig.savefig(output_file, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()