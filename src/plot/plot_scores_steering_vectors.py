# Standard library imports
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import math # For checking nan/inf

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Configure logging (ensure it's configured elsewhere or uncomment)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function to Load Data ---
def load_scored_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads the scored summary data from a JSON file."""
    logger.info(f"Attempting to load scored data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Scored data loaded successfully.")
        # Basic validation
        if 'scored_summaries' not in data or not isinstance(data['scored_summaries'], dict):
            logger.error("Loaded data missing 'scored_summaries' dictionary.")
            return None
        if 'experiment_information' not in data:
            logger.warning("Loaded data missing 'experiment_information'. Plot titles might be generic.")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {file_path} - {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading data: {e}", exc_info=True)
        return None

# --- Data Preparation Function ---
def prepare_plot_data(scored_summaries: Dict[str, Any], metrics_to_extract: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, int]]:
    """
    Extracts specified metrics from scored_summaries, groups by strength,
    and calculates counts.

    Args:
        scored_summaries: The dictionary keyed by article_idx containing scores.
        metrics_to_extract: Dict defining which metrics to pull. 
                           Keys are score categories (e.g., 'sentiment_scores'), 
                           Values are lists of metric names (e.g., ['transformer', 'vader']).

    Returns:
        A tuple containing:
        - data_by_strength: Dict[strength_str, Dict[metric_name, List[float]]]
        - counts_by_strength: Dict[strength_str, int] (number of valid entries per strength)
    """
    
    # Identify and sort steering strengths numerically
    strengths = set()
    for article_scores in scored_summaries.values():
        if isinstance(article_scores, dict):
            strengths.update(article_scores.keys())
            
    try:
        # Sort strengths numerically, assuming they are convertible to float
        sorted_strengths = sorted(list(strengths), key=float)
    except ValueError:
        logger.warning("Could not sort steering strengths numerically, using string sort.")
        sorted_strengths = sorted(list(strengths))

    data_by_strength = defaultdict(lambda: defaultdict(list))
    counts_by_strength = defaultdict(int)
    valid_score_dict_counts = defaultdict(int)

    all_metric_names = [metric for sublist in metrics_to_extract.values() for metric in sublist]

    for article_idx, strength_dict in scored_summaries.items():
        if not isinstance(strength_dict, dict):
            # logger.warning(f"Skipping article {article_idx}: Invalid data format.")
            continue
            
        for strength in sorted_strengths:
            score_dict = strength_dict.get(strength)
            
            if score_dict is not None and isinstance(score_dict, dict):
                 # Increment count if we have a valid score dictionary for this strength/article
                valid_score_dict_counts[strength] += 1
                
                for score_category, metric_names in metrics_to_extract.items():
                    category_scores = score_dict.get(score_category)
                    if category_scores is not None and isinstance(category_scores, dict):
                        for metric in metric_names:
                            value = category_scores.get(metric)
                            # Check if value is a valid number (not None, NaN, Inf)
                            if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                                data_by_strength[strength][metric].append(float(value))
                            else:
                                logger.debug(f"Article {article_idx}, Strength {strength}, Metric {score_category}.{metric}: Invalid or missing value ({value}).")
                    else:
                       logger.debug(f"Article {article_idx}, Strength {strength}: Missing score category '{score_category}'.")
            else:
                 logger.debug(f"Article {article_idx}, Strength {strength}: Missing or invalid score dictionary.")
                 pass # No valid data for this strength/article

    # Use the count of valid score dictionaries for the title count
    counts_by_strength = dict(valid_score_dict_counts)

    return dict(data_by_strength), counts_by_strength, sorted_strengths


# --- Plotting Function ---
def plot_scores_vs_strength(
    scored_data: Dict[str, Any], 
    output_dir: str = "data/plots/sentiment_vectors/",
    base_filename: str = "plot_250"
):
    """
    Generates and saves plots for sentiment and intrinsic scores vs. steering strength.

    Args:
        scored_data: The dictionary containing experiment_info and scored_summaries.
        output_dir: Directory to save the plots.
        base_filename: Base name for the output plot files.
    """
    if not scored_data or 'scored_summaries' not in scored_data:
        logger.error("Invalid or missing scored_data provided for plotting.")
        return
        
    scored_summaries = scored_data['scored_summaries']

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        return

    # --- 1. Plot Sentiment Scores ---
    logger.info("Preparing data for Sentiment plot...")
    sentiment_metrics = {'sentiment_scores': ['transformer', 'vader']}
    sentiment_data, counts, sorted_strengths = prepare_plot_data(scored_summaries, sentiment_metrics)
    
    if not sentiment_data:
         logger.warning("No valid sentiment data found to plot.")
    else:
        logger.info("Generating Sentiment plot...")
        fig_sent, ax1_sent = plt.subplots(figsize=(12, 7))
        ax2_sent = ax1_sent.twinx() # Second y-axis

        colors = {'transformer': 'tab:blue', 'vader': 'tab:red'}
        axes = {'transformer': ax1_sent, 'vader': ax2_sent}
        labels = {'transformer': 'Transformer Sentiment', 'vader': 'VADER Sentiment'}
        
        # Prepare data for plotting (x positions and y values)
        x_positions = np.arange(len(sorted_strengths))
        plot_handles = []
        plot_labels = []

        for metric in ['transformer', 'vader']:
            ax = axes[metric]
            color = colors[metric]
            label = labels[metric]
            
            # Collect x and y for scatter and means
            all_x_scatter = []
            all_y_scatter = []
            y_means = []

            for i, strength in enumerate(sorted_strengths):
                y_values = sentiment_data.get(strength, {}).get(metric, [])
                if y_values:
                    all_x_scatter.extend([x_positions[i]] * len(y_values))
                    all_y_scatter.extend(y_values)
                    y_means.append(np.mean(y_values))
                else:
                    y_means.append(np.nan) # Use NaN if no data for mean line

            # Plot scatter points
            scatter_handle = ax.scatter(all_x_scatter, all_y_scatter, color=color, alpha=0.4, label=f"{label} Points")
            # Plot mean line (connect non-NaN points)
            mean_line_handle, = ax.plot(x_positions, y_means, color='black', marker='o', linestyle='-', label=f"Mean {label}")
            
            plot_handles.extend([scatter_handle, mean_line_handle])
            plot_labels.extend([f"{label} Points", f"Mean {label}"])
            
            ax.set_ylabel(label, color=color)
            ax.tick_params(axis='y', labelcolor=color)

        # Configure combined plot elements
        ax1_sent.set_xlabel("Steering Strength")
        ax1_sent.set_xticks(x_positions)
        ax1_sent.set_xticklabels(sorted_strengths)
        
        # Create title with counts
        count_str = ", ".join([f"{s}: {counts.get(s, 0)}" for s in sorted_strengths])
        plot_title = f"Sentiment Scores vs. Steering Strength\n(Articles per strength: {count_str})"
        ax1_sent.set_title(plot_title, pad=20) # Add padding to avoid overlap

        # Combine legends
        ax1_sent.legend(plot_handles, plot_labels, loc='best')
        
        fig_sent.tight_layout()
        
        # Save plot
        sent_filename = f"{base_filename}_sentiment.png"
        sent_filepath = os.path.join(output_dir, sent_filename)
        try:
            plt.savefig(sent_filepath, dpi=300)
            logger.info(f"Sentiment plot saved to: {sent_filepath}")
        except Exception as e:
            logger.error(f"Failed to save sentiment plot: {e}")
        plt.close(fig_sent) # Close figure to free memory

    # --- 2. Plot Intrinsic Scores ---
    logger.info("Preparing data for Intrinsic Quality plot...")
    intrinsic_metrics = {'intrinsic_scores': ['perplexity', 'distinct_word_2', 'distinct_char_2']}
    intrinsic_data, counts, sorted_strengths = prepare_plot_data(scored_summaries, intrinsic_metrics)

    if not intrinsic_data:
         logger.warning("No valid intrinsic data found to plot.")
    else:
        logger.info("Generating Intrinsic Quality plot...")
        fig_intr, ax1_intr = plt.subplots(figsize=(12, 7))
        ax2_intr = ax1_intr.twinx() # Second y-axis for distinctness

        # Define metrics, axes, colors, labels, markers for means
        metrics_ax1 = ['perplexity']
        metrics_ax2 = ['distinct_word_2', 'distinct_char_2']
        
        colors = {'perplexity': 'tab:green', 'distinct_word_2': 'tab:orange', 'distinct_char_2': 'tab:purple'}
        axes = {'perplexity': ax1_intr, 'distinct_word_2': ax2_intr, 'distinct_char_2': ax2_intr}
        labels = {'perplexity': 'Perplexity', 'distinct_word_2': 'Distinct-2 Words', 'distinct_char_2': 'Distinct-2 Chars'}
        mean_markers = {'perplexity': 'o', 'distinct_word_2': 's', 'distinct_char_2': '^'} # Different markers for means

        x_positions = np.arange(len(sorted_strengths))
        plot_handles = []
        plot_labels = []
        
        all_metrics = metrics_ax1 + metrics_ax2
        for metric in all_metrics:
            ax = axes[metric]
            color = colors[metric]
            label = labels[metric]
            mean_marker = mean_markers[metric]

            # Collect x and y for scatter and means
            all_x_scatter = []
            all_y_scatter = []
            y_means = []
            
            for i, strength in enumerate(sorted_strengths):
                y_values = intrinsic_data.get(strength, {}).get(metric, [])
                if y_values:
                    all_x_scatter.extend([x_positions[i]] * len(y_values))
                    all_y_scatter.extend(y_values)
                    y_means.append(np.mean(y_values))
                else:
                    y_means.append(np.nan)

            # Plot scatter points
            scatter_handle = ax.scatter(all_x_scatter, all_y_scatter, color=color, alpha=0.4, label=f"{label} Points")
            # Plot mean line (connect non-NaN points) - using black as requested
            mean_line_handle, = ax.plot(x_positions, y_means, color='black', marker=mean_marker, linestyle='-', label=f"Mean {label}")

            plot_handles.extend([scatter_handle, mean_line_handle])
            plot_labels.extend([f"{label} Points", f"Mean {label}"])

            # Set axis label only once per axis
            if metric in metrics_ax1:
                 ax.set_ylabel(labels['perplexity'], color=colors['perplexity'])
                 ax.tick_params(axis='y', labelcolor=colors['perplexity'])
            elif metric == metrics_ax2[0]: # Only set label for the first metric on axis 2
                 ax.set_ylabel(f"{labels['distinct_word_2']} / {labels['distinct_char_2']}", color='black') # Shared axis label
                 # Use default color for shared axis ticks, or choose one
                 # ax.tick_params(axis='y', labelcolor='black') 

        # Configure combined plot elements
        ax1_intr.set_xlabel("Steering Strength")
        ax1_intr.set_xticks(x_positions)
        ax1_intr.set_xticklabels(sorted_strengths)
        
        # Create title with counts
        count_str = ", ".join([f"{s}: {counts.get(s, 0)}" for s in sorted_strengths])
        plot_title = f"Intrinsic Quality Scores vs. Steering Strength\n(Articles per strength: {count_str})"
        ax1_intr.set_title(plot_title, pad=20) # Add padding

        # Combine legends
        # Matplotlib might automatically handle colors for labels on ax2, adjust if needed
        ax1_intr.legend(plot_handles, plot_labels, loc='best') 
        
        # Set y-limits if needed, e.g., distinctness scores are usually 0-1
        ax2_intr.set_ylim(bottom=max(0, ax2_intr.get_ylim()[0]), top=min(1, ax2_intr.get_ylim()[1]*1.1)) # Ensure distinct axis is roughly 0-1
        # Maybe adjust perplexity scale if it's very large/small
        # ax1_intr.set_yscale('log') # If perplexity varies greatly

        fig_intr.tight_layout()

        # Save plot
        intr_filename = f"{base_filename}_intrinsic.png"
        intr_filepath = os.path.join(output_dir, intr_filename)
        try:
            plt.savefig(intr_filepath, dpi=300)
            logger.info(f"Intrinsic plot saved to: {intr_filepath}")
        except Exception as e:
            logger.error(f"Failed to save intrinsic plot: {e}")
        plt.close(fig_intr) # Close figure


# --- Example Usage ---
if __name__ == '__main__':
    # Configure logging for example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Replace with the actual path to your *scored* JSON file ---
    scored_json_path = 'data/scores/sentiment_vectors/sentiment_summaries_llama3_1b_NEWTS_train_250_articles_sentiment_sentences_20250426_003933.json'

    # Load the scored data
    scored_data = load_scored_data(scored_json_path)

    # Plot the scores vs. strength
    plot_scores_vs_strength(scored_data)
