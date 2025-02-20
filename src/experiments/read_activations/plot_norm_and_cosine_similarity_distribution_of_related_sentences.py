import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_activations(file_path):
    logging.info(f"Loading activations from {file_path}")
    with open(file_path, 'rb') as f:
        activations = pickle.load(f)
    logging.info(f"Loaded activations with keys: {list(activations.keys())}")
    return activations

def get_min_samples(activations_dict):
    min_samples = float('inf')
    for lang, activations in activations_dict.items():
        for layer, data in activations.items():
            min_samples = min(min_samples, len(data))
            logging.info(f"Language: {lang}, Layer: {layer}, Samples: {len(data)}")
    logging.info(f"Minimum number of samples across all datasets: {min_samples}")
    return min_samples

def extract_samples(activations, num_samples=None, reverse=False):
    if num_samples is None:
        num_samples = min(500, len(activations[list(activations.keys())[0]]))
    logging.info(f"Extracting {num_samples} samples {'in reverse order' if reverse else 'in original order'}")
    
    if reverse:
        extracted = {layer: np.array(activations[layer][-num_samples:])[::-1] for layer in activations}
    else:
        extracted = {layer: np.array(activations[layer][:num_samples]) for layer in activations}
    
    for layer, data in extracted.items():
        logging.info(f"Layer: {layer}, Extracted shape: {data.shape}")
    return extracted

def calculate_contrastive_activations(activations1, activations2):
    logging.info("Calculating contrastive activations")
    contrastive = {}
    for layer in activations1:
        logging.info(f"Layer: {layer}, Shape1: {activations1[layer].shape}, Shape2: {activations2[layer].shape}")
        contrastive[layer] = activations1[layer] - activations2[layer]
    return contrastive

def calculate_cosine_similarities(activations1, activations2):
    logging.info("Calculating cosine similarities")
    similarities = {}
    for layer in activations1:
        logging.info(f"Layer: {layer}, Shape1: {activations1[layer].shape}, Shape2: {activations2[layer].shape}")
        similarities[layer] = cosine_similarity(activations1[layer], activations2[layer]).diagonal()
    return similarities

def plot_distributions(data, plot_type, output_file, num_samples):
    logging.info(f"Plotting {plot_type} distributions for {num_samples} samples")
    comparisons = ['EN+ vs EN-', 'EN+ vs EN+', 'EN+ vs DE+', 'EN+ vs FR+', 'EN+ vs Random', 'Random vs Random']
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    fig.suptitle(f"Distribution of Contrastive Activation {plot_type.capitalize()}s ({num_samples} Samples)", fontsize=16)

    num_layers = len(data[comparisons[0]])
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (num_layers - 1)) for i in range(num_layers)]

    for idx, comparison in enumerate(comparisons):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(comparison)
        ax.set_xlabel(f"Contrastive Activation {plot_type}")
        ax.set_ylabel("Density")

        for layer_idx, layer in enumerate(data[comparison]):
            values = data[comparison][layer]
            logging.info(f"Comparison: {comparison}, Layer: {layer}, Values shape: {values.shape}")
            kde = gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 200)
            ax.plot(x_range, kde(x_range), color=colors[layer_idx], label=f'Layer {layer_idx + 1}')

        if plot_type == 'Cosine Similarity':
            ax.set_xlim(-1, 1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_layers))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_ticks(np.linspace(1, num_layers, num_layers))
        cbar.set_ticklabels(range(1, num_layers + 1))
        cbar.set_label('Layer')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved to {output_file}")

def main():
    input_folder = "data/read_activations/read_sentiment_activations/sentiment_activations/"
    output_folder = "data/read_activations/read_sentiment_activations/sentiment_activations_plots/"
    os.makedirs(output_folder, exist_ok=True)

    languages = ['en', 'de', 'fr', 'random']
    activations_dict = {}

    for lang in languages:
        input_file = os.path.join(input_folder, f"gemma_2b_all_layers_500_sentiment_activations_{lang}.pkl")
        activations_dict[lang] = load_activations(input_file)

    min_samples = get_min_samples(activations_dict)
    logging.info(f"Using {min_samples} samples for each comparison")

    for lang in languages:
        activations_dict[lang] = extract_samples(activations_dict[lang], num_samples=min_samples)

    en_neg = extract_samples(activations_dict['en'], num_samples=min_samples, reverse=True)
    
    # Shift samples for EN+ vs EN+ comparison
    en_pos2 = {}
    for layer in activations_dict['en']:
        en_pos2[layer] = np.roll(activations_dict['en'][layer], 1, axis=0)
    
    # Shuffle random samples
    random2 = {}
    for layer in activations_dict['random']:
        random2[layer] = np.random.permutation(activations_dict['random'][layer])

    logging.info("Calculating contrastive activations")
    contrastive_activations = {
        'EN+ vs EN-': calculate_contrastive_activations(activations_dict['en'], en_neg),
        'EN+ vs EN+': calculate_contrastive_activations(activations_dict['en'], en_pos2),
        'EN+ vs DE+': calculate_contrastive_activations(activations_dict['en'], activations_dict['de']),
        'EN+ vs FR+': calculate_contrastive_activations(activations_dict['en'], activations_dict['fr']),
        'EN+ vs Random': calculate_contrastive_activations(activations_dict['en'], activations_dict['random']),
        'Random vs Random': calculate_contrastive_activations(activations_dict['random'], random2)
    }

    logging.info("Calculating cosine similarities")
    cosine_similarities = {
        'EN+ vs EN-': calculate_cosine_similarities(activations_dict['en'], en_neg),
        'EN+ vs EN+': calculate_cosine_similarities(activations_dict['en'], en_pos2),
        'EN+ vs DE+': calculate_cosine_similarities(activations_dict['en'], activations_dict['de']),
        'EN+ vs FR+': calculate_cosine_similarities(activations_dict['en'], activations_dict['fr']),
        'EN+ vs Random': calculate_cosine_similarities(activations_dict['en'], activations_dict['random']),
        'Random vs Random': calculate_cosine_similarities(activations_dict['random'], random2)
    }

    logging.info("Calculating contrastive activation norms")
    contrastive_norms = {comp: {layer: np.linalg.norm(act, axis=1) for layer, act in layers.items()} 
                         for comp, layers in contrastive_activations.items()}

    plot_distributions(contrastive_norms, 'Norm', os.path.join(output_folder, f"contrastive_activation_norms_distribution_{min_samples}samples.pdf"), min_samples)
    plot_distributions(cosine_similarities, 'Cosine Similarity', os.path.join(output_folder, f"contrastive_activation_cosine_similarities_distribution_{min_samples}samples.pdf"), min_samples)

    logging.info(f"Plots saved to {output_folder}")

if __name__ == "__main__":
    main()