import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_activations(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_positive_samples(activations, num_samples=50):
    # Assuming positive samples are the first half of the dataset
    positive_samples = {layer: activations[layer][:len(activations[layer])//2] for layer in activations}
    return {layer: samples[:num_samples] for layer, samples in positive_samples.items()}

def plot_activations(activations_dict, output_file):
    languages = list(activations_dict.keys())
    layers = list(activations_dict[languages[0]].keys())
    num_layers = len(layers)

    # Calculate grid dimensions
    cols = 3
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6.5 * rows))
    fig.suptitle("Activation Comparison Across Languages and Layers (50 Positive Sentiment Points)", fontsize=16)

    for idx, layer in enumerate(layers):
        ax = axes[idx // cols, idx % cols]
        
        # Combine activations from all languages for this layer
        combined_activations = np.vstack([activations_dict[lang][layer] for lang in languages])
        
        # Perform PCA
        pca = PCA(n_components=2)
        reduced_activations = pca.fit_transform(combined_activations)

        # Plot and label points
        for lang_idx, lang in enumerate(languages):
            start = lang_idx * 50
            end = start + 50
            ax.scatter(reduced_activations[start:end, 0], reduced_activations[start:end, 1], label=lang, s=10)
            for i in range(50):
                ax.annotate(f"{lang}{i+1}", (reduced_activations[start+i, 0], reduced_activations[start+i, 1]), fontsize=4)

        ax.set_title(f"Layer {layer}")
        ax.legend(fontsize='small')
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove any unused subplots
    for idx in range(num_layers, rows * cols):
        fig.delaxes(axes[idx // cols, idx % cols])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    input_folder = "data/read_activations/read_sentiment_activations/sentiment_activations/"
    output_folder = "data/read_activations/read_sentiment_activations/sentiment_activations_plots/"
    os.makedirs(output_folder, exist_ok=True)

    languages = ['en', 'de', 'fr']
    activations_dict = {}

    # Load and extract positive samples for all languages
    for lang in languages:
        input_file = os.path.join(input_folder, f"gemma_2b_all_layers_500_sentiment_activations_{lang}.pkl")
        activations = load_activations(input_file)
        activations_dict[lang] = extract_positive_samples(activations, num_samples=50)

    # Plot and save
    output_file = os.path.join(output_folder, "activation_comparison_all_layers_50_positive_sentiment_3per_row.pdf")
    plot_activations(activations_dict, output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()