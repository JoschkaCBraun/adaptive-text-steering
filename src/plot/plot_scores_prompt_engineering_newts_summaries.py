"""
plot_scores_prompt_engineering_newts_summaries.py

This script generates plots of the scores for the prompt engineering experiments on the NEWTS dataset.

It generates plots for the following:
- Sentiment Scores
- Intrinsic Scores
- Topic Scores
- Extrinsic Scores
- Readability Scores
- Toxicity Scores
"""

# Standard library imports
import os
import math
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGSIZE = (6, 3)
zero_to_one_ylim_with_padding = (-0.05, 1.05)
minus_one_to_one_ylim_with_padding = (-1.1, 1.1)
CONCEPTUAL_CATEGORIES_ORDER = ["Discouraged", "Neutral", "Encouraged"]

# --- Utility Functions ---
def get_nested_value(data_dict: Dict, key_path: str, default: Any = None) -> Any:
    """Safely retrieves a value from a nested dictionary using a dot-separated path."""
    keys = key_path.split('.')
    value = data_dict
    try:
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                logger.debug(f"Cannot traverse further at key '{key}' in path '{key_path}'. Value is not a dict: {value}")
                return default
            if value is None:
                 logger.debug(f"Key '{key}' not found in path '{key_path}'.")
                 return default
        return value
    except Exception as e:
        logger.debug(f"Error accessing nested key '{key_path}': {e}")
        return default

# --- Data Loading and Preparation ---
def load_scored_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads the scored summary data from a JSON file."""
    logger.info(f"Attempting to load scored data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Scored data loaded successfully.")
        
        # Add detailed logging of data structure
        if 'scored_summaries' in data:
            num_articles = len(data['scored_summaries'])
            logger.info(f"Found {num_articles} articles in scored_summaries")
            if num_articles > 0:
                first_article = next(iter(data['scored_summaries'].values()))
                # logger.info(f"Sample article keys: {list(first_article.keys())}")
                # logger.info(f"Sample strategy data structure: {list(first_article.values())[0].keys() if first_article else 'No strategies found'}")
        else:
            logger.error("Loaded data missing 'scored_summaries' dictionary.")
            return None
            
        if 'experiment_information' not in data:
             logger.warning("Loaded data missing 'experiment_information'. Using default plot info.")
             data['experiment_information'] = {}
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
    
def prepare_plot_data(
    scored_summaries: Dict[str, Any],
    metrics_to_extract: List[str],
    ordered_prompt_strategy_keys: List[str], # Actual JSON keys for prompt strategies, defining the x-axis points and order.
    perplexity_metric_path: str = 'intrinsic_scores.perplexity',
    remove_perplexity_outliers: bool = False,
    outlier_percentile: float = 99.0
) -> Tuple[Optional[Dict[str, Dict[str, List[float]]]], Optional[Dict[str, int]], List[str]]:
    """
    Extracts specified metrics from scored_summaries based on ordered_prompt_strategy_keys,
    groups by strategy, calculates counts, and optionally removes perplexity outliers.

    Args:
        scored_summaries: The dictionary keyed by article_idx containing scores.
        metrics_to_extract: List of full metric paths (e.g., 'sentiment_scores.vader',
                           'topic_scores.tid1.dict').
        perplexity_metric_path: The full path to the perplexity metric for outlier removal.
        remove_perplexity_outliers: Whether to remove top percentile of perplexity scores.
        outlier_percentile: Percentile threshold for outlier removal.

    Returns:
        A tuple containing:
        - data_by_strategy_key: Dict[strategy_key, Dict[metric_path, List[float]]]
        - counts_by_strategy_key: Dict[strategy_key, int]
        - ordered_prompt_strategy_keys: List of ordered strategy keys
    """
    # logger.info(f"Preparing plot data for metrics: {metrics_to_extract}")
    # logger.info(f"Using strategy keys: {ordered_prompt_strategy_keys}")
    
    data_by_strategy_key = defaultdict(lambda: defaultdict(list))
    counts_by_strategy_key = defaultdict(int)

    # Log initial data structure
    # logger.info(f"Number of articles in scored_summaries: {len(scored_summaries)}")
    if len(scored_summaries) > 0:
        sample_article = next(iter(scored_summaries.values()))
        # logger.info(f"Sample article strategies: {list(sample_article.keys())}")

    # First pass: collect all perplexity values if needed
    all_perplexity_values = []
    if remove_perplexity_outliers and perplexity_metric_path in metrics_to_extract:
        # logger.info(f"Collecting '{perplexity_metric_path}' values for outlier detection across specified prompt strategies.")
        for article_idx, strategy_scores_dict in scored_summaries.items():
            if not isinstance(strategy_scores_dict, dict): 
                logger.warning(f"Article {article_idx}: Invalid strategy_scores_dict type: {type(strategy_scores_dict)}")
                continue
            for strategy_key in ordered_prompt_strategy_keys:
                score_dict = strategy_scores_dict.get(strategy_key)
                if score_dict and isinstance(score_dict, dict):
                    perplexity_val = get_nested_value(score_dict, perplexity_metric_path)
                    if perplexity_val is not None and isinstance(perplexity_val, (int, float)) and math.isfinite(perplexity_val):
                        all_perplexity_values.append(float(perplexity_val))
                    else:
                        logger.debug(f"Invalid perplexity value for article {article_idx}, strategy {strategy_key}: {perplexity_val}")

    # Calculate perplexity threshold if needed
    perplexity_threshold = None
    if all_perplexity_values and remove_perplexity_outliers:
        try:
            perplexity_threshold = np.percentile(all_perplexity_values, outlier_percentile)
            # logger.info(f"'{perplexity_metric_path}' outlier threshold (top {100 - outlier_percentile:.1f}% of selected strategies): {perplexity_threshold:.2f}")
            # logger.info(f"Number of perplexity values collected: {len(all_perplexity_values)}")
        except IndexError:
             logger.warning(f"Could not calculate percentile for {perplexity_metric_path}. Outlier removal skipped.")
             remove_perplexity_outliers = False

    # Second pass: collect all metric values
    valid_data_found = False
    for article_idx, strategy_scores_dict in scored_summaries.items():
        if not isinstance(strategy_scores_dict, dict):
            logger.debug(f"Article {article_idx}: Invalid or missing strategy_scores_dict.")
            continue

        for strategy_key in ordered_prompt_strategy_keys:
            score_dict = strategy_scores_dict.get(strategy_key)
            if not score_dict or not isinstance(score_dict, dict):
                logger.debug(f"Article {article_idx}, Strategy {strategy_key}: Missing or invalid score dictionary.")
                continue

            counts_by_strategy_key[strategy_key] += 1
            
            for metric_path in metrics_to_extract:
                value = get_nested_value(score_dict, metric_path)
                if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                    # Check for perplexity outliers if needed
                    if (remove_perplexity_outliers and
                        metric_path == perplexity_metric_path and
                        perplexity_threshold is not None and
                        float(value) > perplexity_threshold):
                        logger.debug(f"Filtered out perplexity outlier for {strategy_key}: {value} > {perplexity_threshold}")
                        continue
                    
                    # Add the value to the appropriate list
                    data_by_strategy_key[strategy_key][metric_path].append(float(value))
                    valid_data_found = True
                else:
                    logger.debug(f"Article {article_idx}, Strategy {strategy_key}, Metric {metric_path}: Invalid/missing value ({value}).")

    if not valid_data_found:
        logger.warning("No valid data points found for any metric in the specified prompt strategies.")
        return None, None, ordered_prompt_strategy_keys

    # Log the number of data points collected for each strategy
    # logger.info("Data collection summary:")
    for strategy_key in ordered_prompt_strategy_keys:
        for metric_path in metrics_to_extract:
            num_points = len(data_by_strategy_key[strategy_key][metric_path])
            # logger.info(f"Strategy '{strategy_key}', Metric '{metric_path}': {num_points} data points")
        # logger.info(f"Strategy Key '{strategy_key}': Processed {counts_by_strategy_key.get(strategy_key, 0)} articles.")

    return dict(data_by_strategy_key), dict(counts_by_strategy_key), ordered_prompt_strategy_keys

# --- Core Plotting Function ---
def _create_scatter_mean_plot(
    plot_data: Dict[str, Dict[str, List[float]]],
    counts: Dict[str, int],
    sorted_strengths: List[str],
    plot_config: Dict[str, Any],
    title: str,
    xlabel: str,
    output_filepath: str,
    figsize: Tuple[float, float] = FIGSIZE
):
    """
    Core function to generate a scatter plot with mean lines, supporting dual axes.
    """
    logger.info(f"Generating plot: {title}")
    # logger.info(f"Plot data structure: {list(plot_data.keys())}")
    for strategy in plot_data:
        # logger.info(f"Strategy '{strategy}' metrics: {list(plot_data[strategy].keys())}")
        for metric in plot_data[strategy]:
            values = plot_data[strategy][metric]
            # logger.info(f"Strategy '{strategy}', Metric '{metric}': {len(values)} data points with values: {values}")
    
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = None
    axes = {'primary': ax1}

    # Configure secondary axis if defined
    if plot_config.get('secondary_axis'):
        logger.debug("Configuring secondary Y-axis.")
        ax2 = ax1.twinx()
        axes['secondary'] = ax2

    x_positions = np.arange(len(sorted_strengths))
    all_handles = []
    all_labels = []

    # --- Plot metrics for each axis ---
    for axis_key, ax in axes.items():
        axis_config = plot_config.get(f"{axis_key}_axis")
        if not axis_config or not axis_config.get('metrics'):
            logger.debug(f"No metrics configured for {axis_key} axis.")
            continue

        logger.debug(f"Plotting metrics for {axis_key} axis: {[m['name'] for m in axis_config['metrics']]}")

        for metric_info in axis_config['metrics']:
            metric_path = metric_info['name']
            metric_label = metric_info['label']
            metric_color = metric_info['color']
            scatter_marker = metric_info.get('scatter_marker', 'o') # Default marker
            mean_marker = metric_info.get('mean_marker', 'o')      # Default marker

            # Collect data for scatter and mean calculation
            all_x_scatter = []
            all_y_scatter = []
            y_means = []

            for i, strength in enumerate(sorted_strengths):
                # Use the actual strategy key from plot_data instead of the display label
                strategy_key = list(plot_data.keys())[i]
                y_values = plot_data[strategy_key].get(metric_path, [])
                if y_values:
                    # Log the values being plotted
                    # logger.info(f"Plotting {len(y_values)} values for {strategy_key} - {metric_path}: {y_values}")
                    # Reduce jitter effect for more consistent spacing
                    jitter = np.random.normal(0, 0.02, len(y_values))
                    all_x_scatter.extend([x_positions[i] + jitter])
                    all_y_scatter.extend(y_values)
                    y_means.append(np.mean(y_values))
                    # logger.info(f"Mean value for {strategy_key} - {metric_path}: {y_means[-1]}")
                else:
                    y_means.append(np.nan) # Use NaN if no data for mean line
                    logger.warning(f"No values found for {strategy_key} - {metric_path}")

            # Flatten the list of arrays for scatter plot if jitter was applied
            if all_x_scatter:
                 all_x_scatter = np.concatenate(all_x_scatter)
                 # logger.info(f"Final x positions for scatter: {all_x_scatter}")
                 # logger.info(f"Final y values for scatter: {all_y_scatter}")

            # Plot scatter points
            scatter_handle = ax.scatter(
                all_x_scatter, all_y_scatter,
                color=metric_color,
                alpha=plot_config.get('scatter_alpha', 0.6),
                label=metric_label, # Just use the metric label
                marker=scatter_marker,
                zorder=1 # Scatter behind mean line
            )

            # Plot mean line (connecting non-NaN points)
            valid_indices = ~np.isnan(y_means) # Find where means are valid
            mean_line_handle, = ax.plot(
                x_positions[valid_indices], np.array(y_means)[valid_indices],
                color=plot_config.get('mean_line_color', 'black'),
                marker=mean_marker,
                linestyle=plot_config.get('mean_line_style', '-'),
                label="Mean", # Just use "Mean"
                zorder=10 # Mean line on top with higher z-order
            )
            # logger.info(f"Mean line x positions: {x_positions[valid_indices]}")
            # logger.info(f"Mean line y values: {np.array(y_means)[valid_indices]}")

            # Collect handles/labels for the legend
            all_handles.append(scatter_handle)
            all_labels.append(metric_label)
            all_handles.append(mean_line_handle)
            all_labels.append("Mean")

        # Configure axis labels and limits
        ax.set_ylabel(axis_config['ylabel'], color='black')
        ax.tick_params(axis='y', labelcolor='black')
        if axis_config.get('ylim'):
            ax.set_ylim(axis_config['ylim'])
            logger.debug(f"Set {axis_key} axis ylim: {axis_config['ylim']}")

    # --- Configure overall plot elements ---
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sorted_strengths)
    
    # Set title with extra padding at the top
    ax1.set_title(title, pad=7)
    ax1.grid(axis='y', linestyle='--', alpha=0.7) # Add light grid

    # Set y-ticks for secondary axis to only go up to 1.0
    if ax2 is not None:
        ax2.set_yticks(np.arange(0, 1.1, 0.2))  # Ticks from 0 to 1.0 in steps of 0.2
        if plot_config['secondary_axis'].get('ylim') == (-3.1, 1.2):  # Special case for readability plot
            ax2.set_yticks(np.arange(-3.0, 1.1, 0.5))  # Ticks from -3 to 1.3 in steps of 0.2

    # --- Configure Legend ---
    legend_opts = plot_config.get('legend_opts', {})
    # Use ax1.legend instead of fig.legend to place it inside the plot
    default_legend_opts = {
        'loc': 'upper center', 
        'ncol': 3, 
        'frameon': True,
        'edgecolor': 'black',  # Add border to make it stand out
        'fancybox': True,      # Rounded corners
        'shadow': False,       # No shadow
        'fontsize': 9,         # Increased font size
        'markerscale': 1.2,    # Slightly larger markers in legend
        'columnspacing': 0.5,  # Space between columns
        'handletextpad': 0.5,  # Space between legend handles and labels
        'bbox_to_anchor': (0.107, 0.88),
    }
    final_legend_opts = {**default_legend_opts, **legend_opts} # User opts override defaults

    fig.legend(all_handles, all_labels, **final_legend_opts)
    fig.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create output directory {output_dir}: {e}")
            plt.close(fig)
            return # Cannot save

    # Save plot
    try:
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {output_filepath}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_filepath}: {e}")
    finally:
        plt.close(fig) # Close figure to free memory

# --- Specific Plotting Functions ---

def _get_ordered_keys_and_labels(
    strategy_map: Dict[str, str],
    conceptual_order: List[str],
    custom_display_labels: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """Helper to get ordered actual JSON keys and display labels."""
    ordered_keys = []
    display_labels = []
    
    for i, concept_cat in enumerate(conceptual_order):
        actual_key = strategy_map.get(concept_cat)
        if actual_key:
            ordered_keys.append(actual_key)
            if custom_display_labels and i < len(custom_display_labels):
                display_labels.append(custom_display_labels[i])
            else:
                display_labels.append(concept_cat) # Default to conceptual category
        else:
            # This case should ideally be caught by checking if all strategy_map values are present
            logger.error(f"Conceptual category '{concept_cat}' not found in strategy_map. This indicates a configuration error.")
            # Fallback or raise error, for now, we'll create a placeholder
            ordered_keys.append(f"MISSING_KEY_FOR_{concept_cat}")
            display_labels.append(f"ERR: {concept_cat}")

    if len(ordered_keys) != len(conceptual_order):
        logger.warning(f"Not all conceptual categories ({conceptual_order}) were mapped to actual keys. Check strategy_map.")
        
    return ordered_keys, display_labels


def plot_sentiment_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str, prompt_type: str):
    """Generates plot for Sentiment scores vs. Prompt Strategy."""
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})

    # Use the appropriate strategy map based on prompt type
    if prompt_type == 'sentiment':
        strategy_map = {
            "Discouraged": "sentiment_negative_encouraged",
            "Neutral": "neutral",
            "Encouraged": "sentiment_positive_encouraged"
        }
        display_labels = ["Negative", "Neutral", "Positive"]
    elif prompt_type == 'toxic':
        strategy_map = {
            "Discouraged": "toxicity_avoided",
            "Neutral": "neutral",
            "Encouraged": "toxicity_encouraged"
        }
        display_labels = ["Avoided", "Neutral", "Encouraged"]
    elif prompt_type == 'readability':
        strategy_map = {
            "Discouraged": "readability_complex_encouraged",
            "Neutral": "neutral",
            "Encouraged": "readability_simple_encouraged"
        }
        display_labels = ["Complex", "Neutral", "Simple"]
    else:  # topic
        strategy_map = {
            "Discouraged": "topic_tid2_encouraged",
            "Neutral": "neutral",
            "Encouraged": "topic_tid1_encouraged"
        }
        display_labels = ["TID2", "Neutral", "TID1"]

    ordered_keys, _ = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, display_labels)
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for sentiment plot found in data. Skipping."); return

    metric_paths = ['sentiment_scores.transformer', 'sentiment_scores.vader']
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': 'sentiment_scores.transformer', 'label': 'Transformer', 'color': 'tab:blue', 'mean_marker': 'o', 'scatter_marker': 'o'},
                {'name': 'sentiment_scores.vader', 'label': 'VADER', 'color': 'tab:red', 'mean_marker': '^', 'scatter_marker': '^'}
            ],
            'ylabel': 'Sentiment Score (-1 to 1)',
            'ylim': minus_one_to_one_ylim_with_padding
        },
        'secondary_axis': None,
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 2, 'bbox_to_anchor': (0.125, 0.88)}
    }
    plot_data, counts, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for sentiment plot. Skipping."); return
    _create_scatter_mean_plot(plot_data, counts, display_labels, plot_config,
        title=f"Sentiment Scores vs. {prompt_type.title()} Prompt Strategy",
        xlabel=f"Prompted {prompt_type.title()}",
        output_filepath=os.path.join(output_dir, f"{base_filename}_sentiment.pdf"))


def plot_intrinsic_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str, prompt_type: str, remove_perplexity_outliers: bool = True, outlier_percentile: float = 97.5):
    """Generates plot for Intrinsic Quality scores vs. Prompt Strategy."""
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})

    # Use the appropriate strategy map based on prompt type
    if prompt_type == 'sentiment':
        strategy_map = {
            "Discouraged": "sentiment_negative_encouraged",
            "Neutral": "neutral",
            "Encouraged": "sentiment_positive_encouraged"
        }
        display_labels = ["Negative", "Neutral", "Positive"]
    elif prompt_type == 'toxic':
        strategy_map = {
            "Discouraged": "toxicity_avoided",
            "Neutral": "neutral",
            "Encouraged": "toxicity_encouraged"
        }
        display_labels = ["Avoided", "Neutral", "Encouraged"]
    elif prompt_type == 'readability':
        strategy_map = {
            "Discouraged": "readability_complex_encouraged",
            "Neutral": "neutral",
            "Encouraged": "readability_simple_encouraged"
        }
        display_labels = ["Complex", "Neutral", "Simple"]
    else:  # topic
        strategy_map = {
            "Discouraged": "topic_tid2_encouraged",
            "Neutral": "neutral",
            "Encouraged": "topic_tid1_encouraged"
        }
        display_labels = ["TID2", "Neutral", "TID1"]

    ordered_keys, _ = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, display_labels)
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for intrinsic plot found. Skipping."); return

    metric_paths = ['intrinsic_scores.perplexity', 'intrinsic_scores.distinct_word_2', 'intrinsic_scores.distinct_char_2']
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': 'intrinsic_scores.perplexity', 'label': 'Perplexity', 'color': 'tab:green', 'mean_marker': 'o', 'scatter_marker': 'o'}
            ],
            'ylabel': 'Perplexity',
            'ylim': None
        },
        'secondary_axis': {
             'metrics': [
                 {'name': 'intrinsic_scores.distinct_word_2', 'label': 'Distinct-2 Words', 'color': 'tab:orange', 'mean_marker': 's', 'scatter_marker': 's'},
                 {'name': 'intrinsic_scores.distinct_char_2', 'label': 'Distinct-2 Chars', 'color': 'tab:purple', 'mean_marker': '^', 'scatter_marker': '^'}
             ],
             'ylabel': 'Distinctness Score (0-1)',
             'ylim': (-0.05, 1.4) 
        },
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 3, 'bbox_to_anchor': (0.17, 0.88)}
    }
    plot_data, counts, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys, 'intrinsic_scores.perplexity', remove_perplexity_outliers, outlier_percentile)
    if not plot_data: logger.warning(f"No data for intrinsic plot. Skipping."); return
    _create_scatter_mean_plot(plot_data, counts, display_labels, plot_config,
        title=f"Intrinsic Quality vs. {prompt_type.title()} Prompt Strategy",
        xlabel=f"Prompt Strategy (based on {prompt_type.title()} Prompts)",
        output_filepath=os.path.join(output_dir, f"{base_filename}_intrinsic.pdf"))


def plot_topic_scores(scored_data: Dict[str, Any], topic_id: str, output_dir: str, base_filename: str, prompt_type: str):
    """Generates plot for Topic scores vs. Prompt Strategy for a specific topic ID."""
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})
    alt_topic_id = "tid2" if topic_id == "tid1" else "tid1"

    # Use the appropriate strategy map based on prompt type
    if prompt_type == 'sentiment':
        strategy_map = {
            "Discouraged": "sentiment_negative_encouraged",
            "Neutral": "neutral",
            "Encouraged": "sentiment_positive_encouraged"
        }
        display_labels = ["Negative", "Neutral", "Positive"]
    elif prompt_type == 'toxic':
        strategy_map = {
            "Discouraged": "toxicity_avoided",
            "Neutral": "neutral",
            "Encouraged": "toxicity_encouraged"
        }
        display_labels = ["Avoided", "Neutral", "Encouraged"]
    elif prompt_type == 'readability':
        strategy_map = {
            "Discouraged": "readability_complex_encouraged",
            "Neutral": "neutral",
            "Encouraged": "readability_simple_encouraged"
        }
        display_labels = ["Complex", "Neutral", "Simple"]
    else:  # topic
        strategy_map = {
            "Discouraged": "topic_tid2_encouraged",
            "Neutral": "neutral",
            "Encouraged": "topic_tid1_encouraged"
        }
        display_labels = ["TID2", "Neutral", "TID1"]

    ordered_keys, _ = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, display_labels)
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for topic {topic_id} plot found. Skipping."); return

    metric_paths = [f'topic_scores.{topic_id}.dict', f'topic_scores.{topic_id}.tokenize', f'topic_scores.{topic_id}.lemmatize']
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': f'topic_scores.{topic_id}.dict', 'label': 'Dict Score', 'color': 'tab:cyan', 'mean_marker': 'o', 'scatter_marker': 'o'},
                {'name': f'topic_scores.{topic_id}.tokenize', 'label': 'Tokenize Score', 'color': 'tab:olive', 'mean_marker': 's', 'scatter_marker': 's'},
                {'name': f'topic_scores.{topic_id}.lemmatize', 'label': 'Lemmatize Score', 'color': 'tab:pink', 'mean_marker': '^', 'scatter_marker': '^'}
            ],
            'ylabel': f'Topic Score ({topic_id})',
            'ylim': zero_to_one_ylim_with_padding
        },
        'secondary_axis': None,
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 3}
    }
    plot_data, counts, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for topic {topic_id} plot. Skipping."); return
    _create_scatter_mean_plot(plot_data, counts, display_labels, plot_config,
        title=f"Topic Adherence ({topic_id.upper()}) vs. {prompt_type.title()} Prompt Strategy",
        xlabel=f"Prompt Strategy (based on {prompt_type.title()} Prompts)",
        output_filepath=os.path.join(output_dir, f"{base_filename}_topic_{topic_id}.pdf"))


def plot_extrinsic_scores(scored_data: Dict[str, Any], reference_key: str, output_dir: str, base_filename: str, prompt_type: str):
    """Generates plot for Extrinsic Quality scores vs. Prompt Strategy for a specific reference."""
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})

    # Use the appropriate strategy map based on prompt type
    if prompt_type == 'sentiment':
        strategy_map = {
            "Discouraged": "sentiment_negative_encouraged",
            "Neutral": "neutral",
            "Encouraged": "sentiment_positive_encouraged"
        }
        display_labels = ["Negative", "Neutral", "Positive"]
    elif prompt_type == 'toxic':
        strategy_map = {
            "Discouraged": "toxicity_avoided",
            "Neutral": "neutral",
            "Encouraged": "toxicity_encouraged"
        }
        display_labels = ["Avoided", "Neutral", "Encouraged"]
    elif prompt_type == 'readability':
        strategy_map = {
            "Discouraged": "readability_complex_encouraged",
            "Neutral": "neutral",
            "Encouraged": "readability_simple_encouraged"
        }
        display_labels = ["Complex", "Neutral", "Simple"]
    else:  # topic
        strategy_map = {
            "Discouraged": "topic_tid2_encouraged",
            "Neutral": "neutral",
            "Encouraged": "topic_tid1_encouraged"
        }
        display_labels = ["TID2", "Neutral", "TID1"]

    ordered_keys, _ = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, display_labels)
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for extrinsic plot found. Skipping."); return

    metric_paths = [f'extrinsic_scores.{reference_key}.rouge1', f'extrinsic_scores.{reference_key}.rouge2', f'extrinsic_scores.{reference_key}.rougeL', f'extrinsic_scores.{reference_key}.bert_f1']
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': f'extrinsic_scores.{reference_key}.rouge1', 'label': 'ROUGE-1', 'color': 'tab:blue', 'mean_marker': 'o', 'scatter_marker': 'o'},
                {'name': f'extrinsic_scores.{reference_key}.rouge2', 'label': 'ROUGE-2', 'color': 'tab:red', 'mean_marker': 's', 'scatter_marker': 's'},
                {'name': f'extrinsic_scores.{reference_key}.rougeL', 'label': 'ROUGE-L', 'color': 'tab:green', 'mean_marker': '^', 'scatter_marker': '^'}
            ],
            'ylabel': 'ROUGE Score (0-1)',
            'ylim': zero_to_one_ylim_with_padding
        },
        'secondary_axis': {
             'metrics': [
                 {'name': f'extrinsic_scores.{reference_key}.bert_f1', 'label': 'BERTScore F1', 'color': 'tab:purple', 'mean_marker': 'd', 'scatter_marker': 'd'}
             ],
             'ylabel': 'BERTScore F1 (0-1)',
             'ylim': zero_to_one_ylim_with_padding
        },
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 4, 'bbox_to_anchor': (0.145, 0.88)}
    }
    plot_data, counts, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for extrinsic plot. Skipping."); return
    _create_scatter_mean_plot(plot_data, counts, display_labels, plot_config,
        title=f"Extrinsic Quality vs. {prompt_type.title()} Prompt Strategy",
        xlabel=f"Prompt Strategy (based on {prompt_type.title()} Prompts)",
        output_filepath=os.path.join(output_dir, f"{base_filename}_extrinsic_{reference_key}.pdf"))


def plot_readability_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str, prompt_type: str):
    """Generates plot for Readability scores vs. Prompt Strategy."""
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})

    # Use the appropriate strategy map based on prompt type
    if prompt_type == 'sentiment':
        strategy_map = {
            "Discouraged": "sentiment_negative_encouraged",
            "Neutral": "neutral",
            "Encouraged": "sentiment_positive_encouraged"
        }
        display_labels = ["Negative", "Neutral", "Positive"]
    elif prompt_type == 'toxic':
        strategy_map = {
            "Discouraged": "toxicity_avoided",
            "Neutral": "neutral",
            "Encouraged": "toxicity_encouraged"
        }
        display_labels = ["Avoided", "Neutral", "Encouraged"]
    elif prompt_type == 'readability':
        strategy_map = {
            "Discouraged": "readability_complex_encouraged",
            "Neutral": "neutral",
            "Encouraged": "readability_simple_encouraged"
        }
        display_labels = ["Complex", "Neutral", "Simple"]
    else:  # topic
        strategy_map = {
            "Discouraged": "topic_tid2_encouraged",
            "Neutral": "neutral",
            "Encouraged": "topic_tid1_encouraged"
        }
        display_labels = ["TID2", "Neutral", "TID1"]

    ordered_keys, _ = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, display_labels)
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for readability plot found. Skipping."); return

    metric_paths = ['readability_scores.distilbert', 'readability_scores.deberta']
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': 'readability_scores.deberta', 'label': 'DeBERTa', 'color': 'tab:blue', 'mean_marker': 'o', 'scatter_marker': 'o'}
            ],
            'ylabel': 'DeBERTa Score',
            'ylim': (25.1, -0.1)
        },
        'secondary_axis': {
            'metrics': [
                {'name': 'readability_scores.distilbert', 'label': 'DistilBERT', 'color': 'tab:red', 'mean_marker': '^', 'scatter_marker': '^'}
            ],
            'ylabel': 'DistilBERT Score',
            'ylim': (-3.1, 1.2)
        },
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 2, 'bbox_to_anchor': (0.32, 0.88)}
    }
    plot_data, counts, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for readability plot. Skipping."); return
    _create_scatter_mean_plot(plot_data, counts, display_labels, plot_config,
        title=f"Readability Scores vs. {prompt_type.title()} Prompt Strategy",
        xlabel=f"Prompt Strategy (based on {prompt_type.title()} Prompts)",
        output_filepath=os.path.join(output_dir, f"{base_filename}_readability.pdf"))


def plot_toxicity_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str, prompt_type: str):
    """Generates plot for Toxicity scores vs. Prompt Strategy."""
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})

    # Use the appropriate strategy map based on prompt type
    if prompt_type == 'sentiment':
        strategy_map = {
            "Discouraged": "sentiment_negative_encouraged",
            "Neutral": "neutral",
            "Encouraged": "sentiment_positive_encouraged"
        }
        display_labels = ["Negative", "Neutral", "Positive"]
    elif prompt_type == 'toxic':
        strategy_map = {
            "Discouraged": "toxicity_avoided",
            "Neutral": "neutral",
            "Encouraged": "toxicity_encouraged"
        }
        display_labels = ["Avoided", "Neutral", "Encouraged"]
    elif prompt_type == 'readability':
        strategy_map = {
            "Discouraged": "readability_complex_encouraged",
            "Neutral": "neutral",
            "Encouraged": "readability_simple_encouraged"
        }
        display_labels = ["Complex", "Neutral", "Simple"]
    else:  # topic
        strategy_map = {
            "Discouraged": "topic_tid2_encouraged",
            "Neutral": "neutral",
            "Encouraged": "topic_tid1_encouraged"
        }
        display_labels = ["TID2", "Neutral", "TID1"]

    ordered_keys, _ = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, display_labels)
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for toxicity plot found. Skipping."); return

    metric_paths = [
        'toxicity_scores.toxic_bert',
        'toxicity_scores.severe_toxic_bert',
        'toxicity_scores.roberta_toxicity'
    ]
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': 'toxicity_scores.toxic_bert', 'label': 'Toxic BERT', 'color': 'tab:blue', 'mean_marker': 'o', 'scatter_marker': 'o'},
                {'name': 'toxicity_scores.severe_toxic_bert', 'label': 'Severe Toxic BERT', 'color': 'tab:red', 'mean_marker': 's', 'scatter_marker': 's'},
                {'name': 'toxicity_scores.roberta_toxicity', 'label': 'RoBERTa Toxicity', 'color': 'tab:green', 'mean_marker': '^', 'scatter_marker': '^'}
            ],
            'ylabel': 'Toxicity Score (0-1)',
            'ylim': zero_to_one_ylim_with_padding
        },
        'secondary_axis': None,
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 3, 'bbox_to_anchor': (0.105, 0.88)}
    }
    plot_data, counts, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for toxicity plot. Skipping."); return
    _create_scatter_mean_plot(plot_data, counts, display_labels, plot_config,
        title=f"Toxicity Scores vs. {prompt_type.title()} Prompt Strategy",
        xlabel=f"Prompt Strategy (based on {prompt_type.title()} Prompts)",
        output_filepath=os.path.join(output_dir, f"{base_filename}_toxicity.pdf"))


def main() -> None:
    """
    This function generates plots for the prompt engineering experiments on the NEWTS dataset.
    For each prompt type (toxic, sentiment, readability, topic), it generates plots for ALL score types
    to show how each prompt type affects all metrics.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    FILE_NAME = "prompt_engineering_summaries_llama3_3b_NEWTS_train_250_articles_20250518_030833.json"
    scores_path = os.getenv('SCORES_PATH') 
    plots_path = os.getenv('PLOTS_PATH')
    
    scored_json_path = os.path.join(scores_path, "prompt_engineering", FILE_NAME)
    base_output_dir = os.path.join(plots_path, "newts_summaries")
    base_plot_filename_prefix = FILE_NAME # For plot filenames

    logger.info(f"Input scored JSON: {scored_json_path}")
    logger.info(f"Base output directory: {base_output_dir}")

    scored_data = load_scored_data(scored_json_path)

    if scored_data:
        # Create base directory
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create subdirectories for each prompt type
        prompt_type_dirs = {
            'toxic': os.path.join(base_output_dir, "toxic_prompt"),
            'sentiment': os.path.join(base_output_dir, "sentiment_prompt"),
            'readability': os.path.join(base_output_dir, "readability_prompt"),
            'topic': os.path.join(base_output_dir, "topic_prompt")
        }
        
        for dir_path in prompt_type_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

        logger.info(f"--- Generating Plots for: {base_plot_filename_prefix} ---")

        # For each prompt type, generate all possible score plots
        for prompt_type, output_dir in prompt_type_dirs.items():
            logger.info(f"Generating all score plots for {prompt_type} prompt type")
            
            # 1. Sentiment Scores
            plot_sentiment_scores(
                scored_data=scored_data,
                output_dir=output_dir,
                base_filename=f"{base_plot_filename_prefix}_{prompt_type}_prompt",
                prompt_type=prompt_type
            )

            # 2. Intrinsic Scores
            plot_intrinsic_scores(
                scored_data=scored_data,
                output_dir=output_dir,
                base_filename=f"{base_plot_filename_prefix}_{prompt_type}_prompt",
                prompt_type=prompt_type,
                remove_perplexity_outliers=True,
                outlier_percentile=97.5
            )

            # 3. Topic Scores (for both topic IDs)
            plot_topic_scores(
                scored_data=scored_data,
                topic_id="tid1",
                output_dir=output_dir,
                base_filename=f"{base_plot_filename_prefix}_{prompt_type}_prompt",
                prompt_type=prompt_type
            )
            plot_topic_scores(
                scored_data=scored_data,
                topic_id="tid2",
                output_dir=output_dir,
                base_filename=f"{base_plot_filename_prefix}_{prompt_type}_prompt",
                prompt_type=prompt_type
            )

            # 4. Extrinsic Scores
            plot_extrinsic_scores(
                scored_data=scored_data,
                reference_key="reference_text1",
                output_dir=output_dir,
                base_filename=f"{base_plot_filename_prefix}_{prompt_type}_prompt",
                prompt_type=prompt_type
            )

            # 5. Readability Scores
            plot_readability_scores(
                scored_data=scored_data,
                output_dir=output_dir,
                base_filename=f"{base_plot_filename_prefix}_{prompt_type}_prompt",
                prompt_type=prompt_type
            )

            # 6. Toxicity Scores
            plot_toxicity_scores(
                scored_data=scored_data,
                output_dir=output_dir,
                base_filename=f"{base_plot_filename_prefix}_{prompt_type}_prompt",
                prompt_type=prompt_type
            )

        logger.info(f"--- Plotting Complete for: {base_plot_filename_prefix} ---")
    else:
        logger.error(f"Failed to load scored data from {scored_json_path}. No plots generated.")

# --- Main Execution ---
if __name__ == '__main__':
    main()
