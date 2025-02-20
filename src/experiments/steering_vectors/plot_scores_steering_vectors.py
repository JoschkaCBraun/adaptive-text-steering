from typing import List, Dict
import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from src.utils.load_and_get_utils import get_data_dir

def load_evaluated_reviews(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_language_mismatches(reviews: List[Dict]):
    mismatches = [r for r in reviews if r['evaluation']['detected_language'] != r['prompt_language']]
    mismatch_percentage = len(mismatches) / len(reviews) * 100
    
    print(f"Percentage of language mismatches: {mismatch_percentage:.2f}%")
    print("Up to 5 mismatch examples:")
    for i, review in enumerate(mismatches[:5]):
        print(f"{i+1}. Prompt: {review['prompt_language']}, Detected: {review['evaluation']['detected_language']}")
        print(f"   Review: {review['review'][:100]}...")  # Print first 100 characters
        print()

def plot_perplexity_vs_sentiment_strength(reviews: List[Dict]):
    sentiment_strengths = sorted(set(r['sentiment_strength'] for r in reviews))
    perplexities = {s: [] for s in sentiment_strengths}
    
    for review in reviews:
        perplexities[review['sentiment_strength']].append(review['evaluation']['perplexity'])
    
    means = [np.mean(perplexities[s]) for s in sentiment_strengths]
    errors = [np.std(perplexities[s]) for s in sentiment_strengths]
    
    plt.figure(figsize=(10, 6))
    x_positions = range(len(sentiment_strengths))
    plt.bar(x_positions, means, yerr=errors, capsize=5)
    plt.xlabel('Sentiment Strength')
    plt.xticks(x_positions, [str(s) for s in sentiment_strengths])
    plt.ylabel('Perplexity')
    plt.yscale('log')
    plt.title('Average Perplexity vs Sentiment Strength')
    return plt.gcf()

def plot_sentiment_score_and_confidence(reviews: List[Dict]):
    sentiment_strengths = sorted(set(r['sentiment_strength'] for r in reviews))
    scores = {s: [] for s in sentiment_strengths}
    confidences = {s: [] for s in sentiment_strengths}
    
    for review in reviews:
        scores[review['sentiment_strength']].append(review['evaluation']['sentiment_score'])
        confidences[review['sentiment_strength']].append(review['evaluation']['sentiment_confidence'])
    
    score_means = [np.mean(scores[s]) for s in sentiment_strengths]
    confidence_means = [np.mean(confidences[s]) for s in sentiment_strengths]
    
    x = np.arange(len(sentiment_strengths))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, score_means, width, label='Sentiment Score')
    ax.bar(x + width/2, confidence_means, width, label='Sentiment Confidence')
    
    ax.set_xlabel('Sentiment Strength')
    ax.set_ylabel('Score / Confidence')
    ax.set_title('Average Sentiment Score and Confidence vs Sentiment Strength')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiment_strengths)
    ax.legend()
    
    return fig

def plot_sentiment_score_same_vs_different_language(reviews: List[Dict]):
    sentiment_strengths = sorted(set(r['sentiment_strength'] for r in reviews))
    same_lang = {s: [] for s in sentiment_strengths}
    diff_lang = {s: [] for s in sentiment_strengths}
    
    for review in reviews:
        if review['prompt_language'] == review['vector_language']:
            same_lang[review['sentiment_strength']].append(review['evaluation']['sentiment_score'])
        else:
            diff_lang[review['sentiment_strength']].append(review['evaluation']['sentiment_score'])
    
    same_means = [np.mean(same_lang[s]) for s in sentiment_strengths]
    diff_means = [np.mean(diff_lang[s]) for s in sentiment_strengths]
    
    x = np.arange(len(sentiment_strengths))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, same_means, width, label='Same Language')
    ax.bar(x + width/2, diff_means, width, label='Different Language')
    
    ax.set_xlabel('Sentiment Strength')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Average Sentiment Score: Same vs Different Language')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiment_strengths)
    ax.legend()
    
    return fig

def save_plots(plots: List[tuple], output_dir: str, model_name: str):
    '''Save generated plots to the specified directory with descriptive names.'''
    os.makedirs(output_dir, exist_ok=True)
    
    for plot, name in plots:
        filename = f"{model_name}_{name}.pdf"
        filepath = os.path.join(output_dir, filename)
        plot.savefig(filepath)
        plt.close(plot)
        print(f"Plot saved to {filepath}")

def main():
    sentiment_vectors_data_dir = os.path.join(get_data_dir(os.getcwd()), 'sentiment_vectors_data')
    input_file = os.path.join(sentiment_vectors_data_dir, 'sentiment_vectors_scores',
                              'scores_250_gemma_2b_450.json')
    output_dir = os.path.join(sentiment_vectors_data_dir, 'sentiment_vectors_plots')
    
    reviews = load_evaluated_reviews(input_file)
    
    analyze_language_mismatches(reviews)
    
    model_name = "gemma_2b"  # Extract this from the input file name if needed
    
    plots = [
        (plot_perplexity_vs_sentiment_strength(reviews), "perplexity_vs_sentiment_strength_450"),
        (plot_sentiment_score_and_confidence(reviews), "sentiment_score_and_confidence_450"),
        (plot_sentiment_score_same_vs_different_language(reviews), "sentiment_score_same_vs_different_language_450")
    ]
    
    save_plots(plots, output_dir, model_name)

if __name__ == "__main__":
    main()