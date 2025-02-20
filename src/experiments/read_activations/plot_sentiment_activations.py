import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity

def load_activations(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_sentiment_activations(activations, output_folder, n_pairs=15):
    os.makedirs(output_folder, exist_ok=True)
    
    layers = list(activations['positive'][0].keys())
    num_layers = len(layers)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Average Sentiment Activations Across Layers (First {n_pairs} Pairs)", fontsize=16)
    
    for idx, layer in enumerate(layers):
        ax = axes[idx // 3, idx % 3]
        
        all_avg_activations = []
        colors = []
        labels = []
        
        for sentiment in ['positive', 'negative']:
            for i, sent_activation in enumerate(activations[sentiment][:n_pairs]):
                layer_activation = sent_activation[layer][0].cpu().numpy()
                avg_activation = np.mean(layer_activation, axis=1)  # Shape: (1, 2048)
                
                all_avg_activations.append(avg_activation.flatten())
                colors.append('red' if sentiment == 'positive' else 'blue')
                labels.append(f"{sentiment.capitalize()} {i+1}")
        
        all_avg_activations = np.array(all_avg_activations)
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_activations = pca.fit_transform(all_avg_activations)
        
        # Plot
        scatter = ax.scatter(reduced_activations[:, 0], reduced_activations[:, 1], c=colors, alpha=0.7)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (reduced_activations[i, 0], reduced_activations[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) 
               for c in ['red', 'blue']]
    fig.legend(handles, ['Positive', 'Negative'], loc='lower center', ncol=2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_file = os.path.join(output_folder, f'avg_sentiment_activations_plot*{n_pairs*2}.pdf')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def calculate_average_activation(activation):
    return np.mean(activation.cpu().numpy(), axis=1).flatten()

def calculate_contrastive_activations(positive_activations, negative_activations):
    contrastive_activations = {}
    
    layers = list(positive_activations[0].keys())
    
    for layer in layers:
        contrastive_activations[layer] = []
        for pos, neg in zip(positive_activations, negative_activations):
            pos_avg = calculate_average_activation(pos[layer][0])
            neg_avg = calculate_average_activation(neg[layer][0])
            contrastive = pos_avg - neg_avg
            contrastive_activations[layer].append(contrastive)
    
    return contrastive_activations

def plot_contrastive_activation_norms(contrastive_activations, output_folder):
    layers = list(contrastive_activations.keys())
    num_layers = len(layers)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Distribution of Contrastive Activation Norms Across Layers", fontsize=16)
    
    for idx, layer in enumerate(layers):
        ax = axes[idx // 3, idx % 3]
        
        norms = [np.linalg.norm(act) for act in contrastive_activations[layer]]
        
        # Use kernel density estimation for a smooth distribution
        kde = gaussian_kde(norms)
        x_range = np.linspace(min(norms), max(norms), 100)
        ax.plot(x_range, kde(x_range))
        
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Norm of Contrastive Activation")
        ax.set_ylabel("Density")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = f"{output_folder}/contrastive_activation_norms*100.pdf"
    plt.savefig(output_file)
    print(f"Norms plot saved to {output_file}")

def calculate_mean_contrastive_activation(contrastive_activations):
    """Calculate the mean contrastive activation vector for each layer."""
    mean_activations = {}
    for layer, activations in contrastive_activations.items():
        # Normalize each activation vector
        normalized_activations = [act / np.linalg.norm(act) for act in activations]
        # Calculate the mean
        mean_activations[layer] = np.mean(normalized_activations, axis=0)
        # Normalize the mean vector
        mean_activations[layer] /= np.linalg.norm(mean_activations[layer])
    return mean_activations

def plot_contrastive_activation_similarities(contrastive_activations, output_folder):
    """Plot the density of cosine similarities between contrastive activations and the mean activation."""
    layers = list(contrastive_activations.keys())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("Density of Cosine Similarities to Mean Contrastive Activation Across Layers", fontsize=16)
    
    mean_activations = calculate_mean_contrastive_activation(contrastive_activations)
    
    for layer in layers:
        activations = np.array(contrastive_activations[layer])
        mean_activation = mean_activations[layer]
        
        # Calculate cosine similarities to the mean activation
        similarities = cosine_similarity(activations, [mean_activation]).flatten()
        
        # Calculate the KDE
        kde = gaussian_kde(similarities)
        x_range = np.linspace(-1, 1, 200)
        density = kde(x_range)
        
        # Plot density
        ax.plot(x_range, density, label=f"Layer {layer}", alpha=0.7)
    
    ax.set_xlabel("Cosine Similarity to Mean Contrastive Activation")
    ax.set_ylabel("Density")
    ax.set_xlim(-1, 1)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = f"{output_folder}/contrastive_activation_similarities_density_to_mean*100.pdf"
    plt.savefig(output_file)
    print(f"Similarities density plot saved to {output_file}")

def analyze_sentiment_activations(activations, output_folder):
    contrastive_activations = calculate_contrastive_activations(
        activations['positive'], activations['negative'])
    
    plot_contrastive_activation_norms(contrastive_activations, output_folder)
    plot_contrastive_activation_similarities(contrastive_activations, output_folder)

def main():
    input_file = "data/read_activations/read_sentiment_activations/gemma_2b_sentiment_activations_100.pkl"
    output_folder = "data/read_activations/read_sentiment_activations/sentiment_activations_plots"
    
    activations = load_activations(input_file)
    plot_sentiment_activations(activations, output_folder, n_pairs=15)
    analyze_sentiment_activations(activations, output_folder)

if __name__ == "__main__":
    main()