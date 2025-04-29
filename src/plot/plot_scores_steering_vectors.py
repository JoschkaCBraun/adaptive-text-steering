# Standard library imports
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import math

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

figsize = (6, 3)
base_filename = 'plot_topic_30'

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

# --- Outlier Removal Function ---
def remove_outliers(data: List[float], percentile: float = 99.0) -> List[float]:
    """
    Removes outliers from a list of values based on a percentile threshold.
    
    Args:
        data: List of numeric values
        percentile: Percentile threshold (e.g., 99.0 for top 1%)
        
    Returns:
        List with outliers removed
    """
    if not data or len(data) < 2:
        return data
        
    threshold = np.percentile(data, percentile)
    filtered_data = [x for x in data if x <= threshold]
    
    removed_count = len(data) - len(filtered_data)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} outliers (top {100-percentile:.1f}%) from data")
        
    return filtered_data

# --- Data Preparation Function ---
def prepare_plot_data(scored_summaries: Dict[str, Any], metrics_to_extract: Dict[str, List[str]], 
                     remove_perplexity_outliers: bool = True, outlier_percentile: float = 99.0) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, int], List[str]]:
    """
    Extracts specified metrics from scored_summaries, groups by strength,
    and calculates counts.

    Args:
        scored_summaries: The dictionary keyed by article_idx containing scores.
        metrics_to_extract: Dict defining which metrics to pull. 
                           Keys are score categories (e.g., 'sentiment_scores'), 
                           Values are lists of metric names (e.g., ['transformer', 'vader']).
        remove_perplexity_outliers: Whether to remove top percentile of perplexity scores
        outlier_percentile: Percentile threshold for outlier removal (e.g., 99.0 for top 1%)

    Returns:
        A tuple containing:
        - data_by_strength: Dict[strength_str, Dict[metric_name, List[float]]]
        - counts_by_strength: Dict[strength_str, int] (number of valid entries per strength)
        - sorted_strengths: List of sorted strength values
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

    # First pass: collect all perplexity values for outlier detection if needed
    all_perplexity_values = []
    if remove_perplexity_outliers and 'intrinsic_scores' in metrics_to_extract and 'perplexity' in metrics_to_extract['intrinsic_scores']:
        for article_idx, strength_dict in scored_summaries.items():
            if not isinstance(strength_dict, dict):
                continue
                
            for strength in sorted_strengths:
                score_dict = strength_dict.get(strength)
                
                if score_dict is not None and isinstance(score_dict, dict):
                    intrinsic_scores = score_dict.get('intrinsic_scores')
                    if intrinsic_scores is not None and isinstance(intrinsic_scores, dict):
                        perplexity = intrinsic_scores.get('perplexity')
                        if perplexity is not None and isinstance(perplexity, (int, float)) and math.isfinite(perplexity):
                            all_perplexity_values.append(float(perplexity))
    
    # Calculate perplexity threshold if needed
    perplexity_threshold = None
    if all_perplexity_values and remove_perplexity_outliers:
        perplexity_threshold = np.percentile(all_perplexity_values, outlier_percentile)
        logger.info(f"Perplexity outlier threshold (top {100-outlier_percentile:.1f}%): {perplexity_threshold:.2f}")

    # Second pass: collect data with outlier filtering
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
                                # Apply outlier filtering for perplexity if enabled
                                if (remove_perplexity_outliers and 
                                    score_category == 'intrinsic_scores' and 
                                    metric == 'perplexity' and 
                                    perplexity_threshold is not None and 
                                    float(value) > perplexity_threshold):
                                    logger.debug(f"Filtered out perplexity outlier: {value} > {perplexity_threshold}")
                                    continue
                                    
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
    base_filename: str = "plot_30",
    output_dir: str = "data/plots/sentiment_vectors/",
    remove_perplexity_outliers: bool = True,
    outlier_percentile: float = 99.0
):
    """
    Generates and saves plots for sentiment and intrinsic scores vs. steering strength.

    Args:
        scored_data: The dictionary containing experiment_info and scored_summaries.
        output_dir: Directory to save the plots.
        base_filename: Base name for the output plot files.
        remove_perplexity_outliers: Whether to remove top percentile of perplexity scores
        outlier_percentile: Percentile threshold for outlier removal (e.g., 99.0 for top 1%)
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
    sentiment_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries, 
        sentiment_metrics,
        remove_perplexity_outliers=False  # No outlier removal for sentiment
    )
    
    if not sentiment_data:
         logger.warning("No valid sentiment data found to plot.")
    else:
        logger.info("Generating Sentiment plot...")
        fig_sent, ax_sent = plt.subplots(figsize=figsize)

        colors = {'transformer': 'tab:blue', 'vader': 'tab:red'}
        labels = {'transformer': 'Transformer Sentiment', 'vader': 'VADER Sentiment'}
        scatter_markers = {'transformer': 'o', 'vader': '^'}  # Define scatter markers for sentiment
        mean_markers = {'transformer': 'o', 'vader': '^'}  # Define mean markers for sentiment
        
        # Prepare data for plotting (x positions and y values)
        x_positions = np.arange(len(sorted_strengths))
        plot_handles = []
        plot_labels = []

        for metric in ['transformer', 'vader']:
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
            scatter_handle = ax_sent.scatter(all_x_scatter, all_y_scatter, color=color, alpha=0.4, label=f"{label} Points", marker=scatter_markers[metric])
            # Plot mean line (connect non-NaN points)
            mean_line_handle, = ax_sent.plot(x_positions, y_means, color='black', marker=mean_markers[metric], linestyle='-', label=f"Mean {label}")
            
            plot_handles.extend([scatter_handle, mean_line_handle])
            plot_labels.extend([f"{label} Points", f"Mean {label}"])

        # Configure plot elements
        ax_sent.set_xlabel("Steering Strength")
        ax_sent.set_xticks(x_positions)
        ax_sent.set_xticklabels(sorted_strengths)
        ax_sent.set_ylabel("Sentiment (-1 negative, 0 neutral, 1 positive)")
        ax_sent.set_ylim(-1.05, 1.05)
        
        plot_title = f"Sentiment Scores vs. Steering Strength"
        ax_sent.set_title(plot_title)

        # Add legend at bottom right with 2 columns
        ax_sent.legend(plot_handles, plot_labels, loc='lower right', ncol=2, bbox_to_anchor=(1.0, 0.0), frameon=True, facecolor='white', edgecolor='gray')
        
        fig_sent.tight_layout()
        
        # Save plot
        sent_filename = f"{base_filename}_sentiment.pdf"
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
    intrinsic_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries, 
        intrinsic_metrics,
        remove_perplexity_outliers=remove_perplexity_outliers,
        outlier_percentile=outlier_percentile
    )

    if not intrinsic_data:
         logger.warning("No valid intrinsic data found to plot.")
    else:
        logger.info("Generating Intrinsic Quality plot...")
        fig_intr, ax1_intr = plt.subplots(figsize=figsize)
        ax2_intr = ax1_intr.twinx() # Second y-axis for distinctness

        # Define metrics, axes, colors, labels, markers for means
        metrics_ax1 = ['perplexity']
        metrics_ax2 = ['distinct_word_2', 'distinct_char_2']
        
        colors = {'perplexity': 'tab:green', 'distinct_word_2': 'tab:orange', 'distinct_char_2': 'tab:purple'}
        axes = {'perplexity': ax1_intr, 'distinct_word_2': ax2_intr, 'distinct_char_2': ax2_intr}
        labels = {'perplexity': 'Perplexity', 'distinct_word_2': 'Distinct-2 Words', 'distinct_char_2': 'Distinct-2 Chars'}
        mean_markers = {'perplexity': 'o', 'distinct_word_2': 's', 'distinct_char_2': '^'} # Different markers for means
        scatter_markers = {'transformer': 'o', 'vader': '^'}  # Define scatter markers for sentiment

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
            scatter_handle = ax.scatter(all_x_scatter, all_y_scatter, color=color, alpha=0.4, label=f"{label} Points", zorder=1)
            # Plot mean line (connect non-NaN points) - using black as requested
            mean_line_handle, = ax.plot(x_positions, y_means, color='black', marker=mean_marker, linestyle='-', label=f"Mean {label}", zorder=2)

            plot_handles.extend([scatter_handle, mean_line_handle])
            plot_labels.extend([f"{label} Points", f"Mean {label}"])

            # Set axis label only once per axis
            if metric in metrics_ax1:
                 ax.set_ylabel(labels['perplexity'], color=colors['perplexity'])
                 ax.tick_params(axis='y', labelcolor=colors['perplexity'])
            elif metric == metrics_ax2[0]: # Only set label for the first metric on axis 2
                 ax.set_ylabel(f"{labels['distinct_word_2']} / {labels['distinct_char_2']}", color='black') # Shared axis label

        # Configure combined plot elements
        ax1_intr.set_xlabel("Steering Strength")
        ax1_intr.set_xticks(x_positions)
        ax1_intr.set_xticklabels(sorted_strengths)
        
        plot_title = f"Intrinsic Quality Scores vs. Steering Strength"
        ax1_intr.set_title(plot_title)

        # Set y-limits if needed, e.g., distinctness scores are usually 0-1
        ax2_intr.set_ylim(bottom=0, top=1.05) # Set distinct axis to 0-1.05
            # Add legend at bottom right with 3 columns, similar to sentiment plot
        fig_intr.legend(plot_handles, 
                        plot_labels, 
                        loc='upper center',
                        bbox_to_anchor=(0.5, 0.90),
                        ncol=3, 
                        frameon=True)


        fig_intr.tight_layout()

        # Save plot
        intr_filename = f"{base_filename}_intrinsic.pdf"
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
    scored_json_path = 'data/scores/topic_vectors/topic_summaries_llama3_1b_NEWTS_train_30_articles_topic_words_20250428_170346.json'

    # Load the scored data
    scored_data = load_scored_data(scored_json_path)

    # Plot the scores vs. strength with outlier removal
    plot_scores_vs_strength(scored_data, remove_perplexity_outliers=True, outlier_percentile=99.0)
