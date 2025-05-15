"""
plot_scores_steered_and_prompted_newts_summaries.py

This script generates plots of the scores for the steered and prompted experiments on the NEWTS dataset.

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGSIZE = (6, 3)
zero_to_one_ylim_with_padding = (-0.05, 1.05)
minus_one_to_one_ylim_with_padding = (-1.1, 1.1)
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
                # Handle cases where an intermediate key might point to a list or non-dict
                logger.debug(f"Cannot traverse further at key '{key}' in path '{key_path}'. Value is not a dict: {value}")
                return default
            if value is None: # Stop if any key is missing
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
        # Basic validation
        if 'scored_summaries' not in data or not isinstance(data['scored_summaries'], dict):
            logger.error("Loaded data missing 'scored_summaries' dictionary.")
            return None
        # Allow missing experiment_information with a warning
        if 'experiment_information' not in data:
             logger.warning("Loaded data missing 'experiment_information'. Using default plot info.")
             data['experiment_information'] = {} # Add empty dict to avoid errors later

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
    metrics_to_extract: List[str], # Now a list of full metric paths
    perplexity_metric_path: str = 'intrinsic_scores.perplexity', # Full path for perplexity
    remove_perplexity_outliers: bool = False,
    outlier_percentile: float = 99.0
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, int], List[str]]:
    """
    Extracts specified metrics (using full dot.notation paths) from scored_summaries,
    groups by strength, calculates counts, and optionally removes perplexity outliers.

    Args:
        scored_summaries: The dictionary keyed by article_idx containing scores.
        metrics_to_extract: List of full metric paths (e.g., 'sentiment_scores.vader',
                           'topic_scores.tid1.dict').
        perplexity_metric_path: The full path to the perplexity metric for outlier removal.
        remove_perplexity_outliers: Whether to remove top percentile of perplexity scores.
        outlier_percentile: Percentile threshold for outlier removal.

    Returns:
        A tuple containing:
        - data_by_strength: Dict[strength_str, Dict[full_metric_path, List[float]]]
        - counts_by_strength: Dict[strength_str, int] (number of valid entries per strength)
        - sorted_strengths: List of sorted strength values
    """
    # Identify and sort steering strengths numerically
    strengths = set()
    for article_scores in scored_summaries.values():
        if isinstance(article_scores, dict):
            strengths.update(article_scores.keys())

    try:
        sorted_strengths = sorted([s for s in strengths if s is not None], key=float)
    except ValueError:
        logger.warning("Could not sort steering strengths numerically, using string sort.")
        sorted_strengths = sorted([s for s in strengths if s is not None])

    data_by_strength = defaultdict(lambda: defaultdict(list))
    counts_by_strength = defaultdict(int)
    valid_score_dict_counts = defaultdict(int)

    # --- First pass: Collect all perplexity values for outlier detection if needed ---
    all_perplexity_values = []
    if remove_perplexity_outliers and perplexity_metric_path in metrics_to_extract:
        logger.info(f"Collecting '{perplexity_metric_path}' values for outlier detection.")
        for article_idx, strength_dict in scored_summaries.items():
            if not isinstance(strength_dict, dict): continue
            for strength in sorted_strengths:
                score_dict = strength_dict.get(strength)
                if score_dict is not None and isinstance(score_dict, dict):
                    perplexity_val = get_nested_value(score_dict, perplexity_metric_path)
                    if perplexity_val is not None and isinstance(perplexity_val, (int, float)) and math.isfinite(perplexity_val):
                        all_perplexity_values.append(float(perplexity_val))

    # Calculate perplexity threshold if needed
    perplexity_threshold = None
    if all_perplexity_values and remove_perplexity_outliers:
        try:
            perplexity_threshold = np.percentile(all_perplexity_values, outlier_percentile)
            logger.info(f"'{perplexity_metric_path}' outlier threshold (top {100 - outlier_percentile:.1f}%): {perplexity_threshold:.2f}")
        except IndexError:
             logger.warning(f"Could not calculate percentile for {perplexity_metric_path}. Outlier removal skipped.")
             remove_perplexity_outliers = False # Disable if calculation failed


    # --- Second pass: Collect data with potential outlier filtering ---
    for article_idx, strength_dict in scored_summaries.items():
        if not isinstance(strength_dict, dict):
            continue

        for strength in sorted_strengths:
            score_dict = strength_dict.get(strength)

            if score_dict is not None and isinstance(score_dict, dict):
                valid_score_dict_counts[strength] += 1 # Count this strength entry

                for metric_path in metrics_to_extract:
                    value = get_nested_value(score_dict, metric_path)

                    # Check if value is a valid number (not None, NaN, Inf)
                    if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                        # Apply outlier filtering for perplexity if enabled
                        if (remove_perplexity_outliers and
                            metric_path == perplexity_metric_path and
                            perplexity_threshold is not None and
                            float(value) > perplexity_threshold):
                            logger.debug(f"Filtered out perplexity outlier: {value} > {perplexity_threshold}")
                            continue # Skip adding this outlier

                        data_by_strength[strength][metric_path].append(float(value))
                    else:
                        logger.debug(f"Article {article_idx}, Strength {strength}, Metric {metric_path}: Invalid or missing value ({value}).")
            else:
                 logger.debug(f"Article {article_idx}, Strength {strength}: Missing or invalid score dictionary.")

    # Use the count of valid score dictionaries
    counts_by_strength = dict(valid_score_dict_counts)

    # Log counts for verification
    for strength in sorted_strengths:
        logger.debug(f"Strength {strength}: Found {counts_by_strength.get(strength, 0)} valid score entries.")
        # for metric_path in metrics_to_extract:
        #      logger.debug(f"  Metric {metric_path}: Collected {len(data_by_strength.get(strength, {}).get(metric_path, []))} values.")


    return dict(data_by_strength), counts_by_strength, sorted_strengths


# --- Core Plotting Function ---

def _format_x_tick_label(strength: str) -> str:
    """Format x-axis tick label to include D/N/E suffix based on steering strength value."""
    try:
        strength_float = float(strength)
        if strength_float < 0:
            return f"{strength}/D"
        elif strength_float > 0:
            return f"{strength}/E"
        else:
            return f"{strength}/N"
    except ValueError:
        # If strength can't be converted to float, return as is
        return strength

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

    Args:
        plot_data: Data grouped by strength and metric path.
        counts: Counts of valid entries per strength.
        sorted_strengths: List of sorted strength values.
        plot_config: Dictionary defining metrics, axes, labels, colors, etc.
                     Expected structure:
                     {
                         'primary_axis': {
                             'metrics': [{'name': str, 'label': str, 'color': str, 'mean_marker': str, 'scatter_marker': str}, ...],
                             'ylabel': str,
                             'ylim': Optional[Tuple[float, float]]
                         },
                         'secondary_axis': Optional[{... similar structure ...}],
                         'scatter_alpha': float,
                         'mean_line_color': str,
                         'mean_line_style': str,
                         'legend_opts': Dict[str, Any]
                     }
        title: The main title for the plot.
        xlabel: Label for the x-axis.
        output_filepath: Full path to save the plot file.
        FIGSIZE: Figure size tuple.
    """
    logger.info(f"Generating plot: {title}")
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
                y_values = plot_data.get(strength, {}).get(metric_path, [])
                if y_values:
                    # Jitter x-positions slightly for scatter visibility
                    jitter = np.random.normal(0, 0.05, len(y_values))
                    all_x_scatter.extend([x_positions[i] + jitter])
                    all_y_scatter.extend(y_values)
                    y_means.append(np.mean(y_values))
                else:
                    y_means.append(np.nan) # Use NaN if no data for mean line


            # Flatten the list of arrays for scatter plot if jitter was applied
            if all_x_scatter:
                 all_x_scatter = np.concatenate(all_x_scatter)


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
    ax1.tick_params(axis='x', labelsize=8.5)    # make xticks smaller
    ax1.set_xlabel(xlabel)
    ax1.set_xticks(x_positions)
    # Format x-tick labels to include D/N/E suffix
    formatted_labels = [_format_x_tick_label(strength) for strength in sorted_strengths]
    ax1.set_xticklabels(formatted_labels)
    
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

def plot_sentiment_scores(
    scored_data: Dict[str, Any],
    output_dir: str = "data/plots/sentiment_vectors/",
    base_filename: str = "plot_30"
):
    """Generates plot for Sentiment scores vs. Steering Strength."""
    if 'scored_summaries' not in scored_data: return
    experiment_info = scored_data.get('experiment_information', {})
    behavior_type = experiment_info.get('behavior_type', 'Unknown').capitalize()
    plot_title_prefix = experiment_info.get('plot_title_prefix', 'Sentiment Scores') # Get prefix from data or use default

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
        'legend_opts': {'loc': 'upper left', 'ncol': 2, 'bbox_to_anchor': (0.125, 0.88)} # Below title
    }

    plot_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries=scored_data['scored_summaries'],
        metrics_to_extract=metric_paths,
        remove_perplexity_outliers=False # No outlier removal for sentiment
    )

    if not plot_data:
        logger.warning("No valid sentiment data found to plot.")
        return

    output_filepath = os.path.join(output_dir, f"{base_filename}_sentiment.pdf")
    _create_scatter_mean_plot(
        plot_data=plot_data,
        counts=counts,
        sorted_strengths=sorted_strengths,
        plot_config=plot_config,
        title=f"{plot_title_prefix} for {behavior_type} Steering and Prompting",
        xlabel="Steering Strength / Aligned Prompt Type (Discourage, Neutral, Encourage)",
        output_filepath=output_filepath,
        figsize=FIGSIZE
    )

def plot_intrinsic_scores(
    scored_data: Dict[str, Any],
    output_dir: str = "data/plots/sentiment_vectors/",
    base_filename: str = "plot_30",
    remove_perplexity_outliers: bool = False,
    outlier_percentile: float = 99.0
):
    """Generates plot for Intrinsic Quality scores vs. Steering Strength."""
    if 'scored_summaries' not in scored_data: return
    experiment_info = scored_data.get('experiment_information', {})
    behavior_type = experiment_info.get('behavior_type', 'Unknown').capitalize()
    plot_title_prefix = experiment_info.get('plot_title_prefix', 'Intrinsic Quality') # Get prefix from data or use default

    metric_paths = [
        'intrinsic_scores.perplexity',
        'intrinsic_scores.distinct_word_2',
        'intrinsic_scores.distinct_char_2'
    ]
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': 'intrinsic_scores.perplexity', 'label': 'Perplexity', 'color': 'tab:green', 'mean_marker': 'o', 'scatter_marker': 'o'}
            ],
            'ylabel': 'Perplexity',
            'ylim': None # Auto-scale perplexity unless outliers are extreme
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
        'legend_opts': {'loc': 'upper left', 'ncol': 3, 'bbox_to_anchor': (0.17, 0.88)} # Below title
    }

    plot_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries=scored_data['scored_summaries'],
        metrics_to_extract=metric_paths,
        perplexity_metric_path='intrinsic_scores.perplexity', # Specify which metric is perplexity
        remove_perplexity_outliers=remove_perplexity_outliers,
        outlier_percentile=outlier_percentile
    )

    if not plot_data:
        logger.warning("No valid intrinsic data found to plot.")
        return

    output_filepath = os.path.join(output_dir, f"{base_filename}_intrinsic.pdf")
    _create_scatter_mean_plot(
        plot_data=plot_data,
        counts=counts,
        sorted_strengths=sorted_strengths,
        plot_config=plot_config,
        title=f"{plot_title_prefix} for {behavior_type} Steering and Prompting",
        xlabel="Steering Strength / Aligned Prompt Type (Discourage, Neutral, Encourage)",
        output_filepath=output_filepath,
        figsize=FIGSIZE
    )

def plot_topic_scores(
    scored_data: Dict[str, Any],
    topic_id: str = "tid1", # Which topic ID to plot (e.g., 'tid1')
    output_dir: str = "data/plots/sentiment_vectors/",
    base_filename: str = "plot_30"
):
    """Generates plot for Topic scores vs. Steering Strength for a specific topic ID."""
    if 'scored_summaries' not in scored_data: return
    experiment_info = scored_data.get('experiment_information', {})
    behavior_type = experiment_info.get('behavior_type', 'Unknown').capitalize()
    plot_title_prefix = experiment_info.get('plot_title_prefix', 'Topic Scores') # Get prefix from data or use default

    # Dynamically create metric paths based on topic_id
    metric_paths = [
        f'topic_scores.{topic_id}.dict',
        f'topic_scores.{topic_id}.tokenize',
        f'topic_scores.{topic_id}.lemmatize'
    ]
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
        'legend_opts': {'loc': 'upper left', 'ncol': 3} # Below title
    }

    plot_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries=scored_data['scored_summaries'],
        metrics_to_extract=metric_paths,
        remove_perplexity_outliers=False # No outlier removal needed here
    )

    # Check if *any* topic data was found
    data_found = any(
        metric_path in plot_data.get(strength, {})
        for strength in sorted_strengths
        for metric_path in metric_paths
    )

    if not data_found:
        logger.warning(f"No valid topic data found for {topic_id} to plot.")
        return

    output_filepath = os.path.join(output_dir, f"{base_filename}_topic_{topic_id}.pdf")
    _create_scatter_mean_plot(
        plot_data=plot_data,
        counts=counts,
        sorted_strengths=sorted_strengths,
        plot_config=plot_config,
        title=f"{plot_title_prefix} for {behavior_type} Steering and Prompting",
        xlabel="Steering Strength / Aligned Prompt Type (Discourage, Neutral, Encourage)",
        output_filepath=output_filepath,
        figsize=FIGSIZE
    )

def plot_extrinsic_scores(
    scored_data: Dict[str, Any],
    reference_key: str = "reference_text1", # Which reference text results to plot
    output_dir: str = "data/plots/sentiment_vectors/",
    base_filename: str = "plot_30"
):
    """Generates plot for Extrinsic Quality scores vs. Steering Strength for a specific reference."""
    if 'scored_summaries' not in scored_data: return
    experiment_info = scored_data.get('experiment_information', {})
    behavior_type = experiment_info.get('behavior_type', 'Unknown').capitalize()
    plot_title_prefix = experiment_info.get('plot_title_prefix', 'Extrinsic Quality') # Get prefix from data or use default

    # Dynamically create metric paths based on reference_key
    metric_paths = [
        f'extrinsic_scores.{reference_key}.rouge1',
        f'extrinsic_scores.{reference_key}.rouge2',
        f'extrinsic_scores.{reference_key}.rougeL',
        f'extrinsic_scores.{reference_key}.bert_f1'
    ]
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': f'extrinsic_scores.{reference_key}.rouge1', 'label': 'ROUGE-1', 'color': 'tab:blue', 'mean_marker': 'o', 'scatter_marker': 'o'},
                {'name': f'extrinsic_scores.{reference_key}.rouge2', 'label': 'ROUGE-2', 'color': 'tab:red', 'mean_marker': 's', 'scatter_marker': 's'},
                {'name': f'extrinsic_scores.{reference_key}.rougeL', 'label': 'ROUGE-L', 'color': 'tab:green', 'mean_marker': '^', 'scatter_marker': '^'}
            ],
            'ylabel': 'ROUGE Score (0-1)',
            'ylim': zero_to_one_ylim_with_padding # ROUGE scores are typically 0-1
        },
        'secondary_axis': {
             'metrics': [
                 {'name': f'extrinsic_scores.{reference_key}.bert_f1', 'label': 'BERTScore F1', 'color': 'tab:purple', 'mean_marker': 'd', 'scatter_marker': 'd'}
             ],
             'ylabel': 'BERTScore F1 (0-1)',
             'ylim': zero_to_one_ylim_with_padding # BERTScore F1 is also typically 0-1
        },
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 4, 'bbox_to_anchor': (0.145, 0.88)} # Below title
    }

    plot_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries=scored_data['scored_summaries'],
        metrics_to_extract=metric_paths,
        remove_perplexity_outliers=False # No outlier removal needed here
    )

    # Check if *any* extrinsic data was found for this reference
    data_found = any(
        metric_path in plot_data.get(strength, {})
        for strength in sorted_strengths
        for metric_path in metric_paths
    )

    if not data_found:
        logger.warning(f"No valid extrinsic data found for reference '{reference_key}' to plot.")
        return

    output_filepath = os.path.join(output_dir, f"{base_filename}_extrinsic_{reference_key}.pdf")
    _create_scatter_mean_plot(
        plot_data=plot_data,
        counts=counts,
        sorted_strengths=sorted_strengths,
        plot_config=plot_config,
        title=f"{plot_title_prefix} for {behavior_type} Steering and Prompting",
        xlabel="Steering Strength / Aligned Prompt Type (Discourage, Neutral, Encourage)",
        output_filepath=output_filepath,
        figsize=FIGSIZE
    )

def plot_readability_scores(
    scored_data: Dict[str, Any],
    output_dir: str = "data/plots/sentiment_vectors/",
    base_filename: str = "plot_30"
):
    """Generates plot for Readability scores vs. Steering Strength."""
    if 'scored_summaries' not in scored_data: return
    experiment_info = scored_data.get('experiment_information', {})
    behavior_type = experiment_info.get('behavior_type', 'Unknown').capitalize()
    plot_title_prefix = experiment_info.get('plot_title_prefix', 'Readability Scores') # Get prefix from data or use default

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
            'ylim': (-3.1, 1.2) # Reversed axis for DistilBERT
        },
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 2, 'bbox_to_anchor': (0.32, 0.88)} # Below title
    }

    plot_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries=scored_data['scored_summaries'],
        metrics_to_extract=metric_paths,
        remove_perplexity_outliers=False # No outlier removal needed here
    )

    if not plot_data:
        logger.warning("No valid readability data found to plot.")
        return

    output_filepath = os.path.join(output_dir, f"{base_filename}_readability.pdf")
    _create_scatter_mean_plot(
        plot_data=plot_data,
        counts=counts,
        sorted_strengths=sorted_strengths,
        plot_config=plot_config,
        title=f"{plot_title_prefix} for {behavior_type} Steering and Prompting",
        xlabel="Steering Strength / Aligned Prompt Type (Discourage, Neutral, Encourage)",
        output_filepath=output_filepath,
        figsize=FIGSIZE
    )

def plot_toxicity_scores(
    scored_data: Dict[str, Any],
    output_dir: str = "data/plots/sentiment_vectors/",
    base_filename: str = "plot_30"
):
    """Generates plot for Toxicity scores vs. Steering Strength."""
    if 'scored_summaries' not in scored_data: return
    experiment_info = scored_data.get('experiment_information', {})
    behavior_type = experiment_info.get('behavior_type', 'Unknown').capitalize()
    plot_title_prefix = experiment_info.get('plot_title_prefix', 'Toxicity Scores') # Get prefix from data or use default

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
            'ylim': zero_to_one_ylim_with_padding # Toxicity scores are typically 0-1
        },
        'secondary_axis': None,
        'scatter_alpha': 0.6,
        'mean_line_color': 'black',
        'mean_line_style': '-',
        'legend_opts': {'loc': 'upper left', 'ncol': 3, 'bbox_to_anchor': (0.105, 0.88)} # Below title
    }

    plot_data, counts, sorted_strengths = prepare_plot_data(
        scored_summaries=scored_data['scored_summaries'],
        metrics_to_extract=metric_paths,
        remove_perplexity_outliers=False # No outlier removal needed here
    )

    if not plot_data:
        logger.warning("No valid toxicity data found to plot.")
        return

    output_filepath = os.path.join(output_dir, f"{base_filename}_toxicity.pdf")
    _create_scatter_mean_plot(
        plot_data=plot_data,
        counts=counts,
        sorted_strengths=sorted_strengths,
        plot_config=plot_config,
        title=f"{plot_title_prefix} for {behavior_type} Steering and Prompting",
        xlabel="Steering Strength / Aligned Prompt Type (Discourage, Neutral, Encourage)",
        output_filepath=output_filepath,
        figsize=FIGSIZE
    )

def main() -> None:
    # Configure logging for example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # --- Replace with the actual path to your *scored* JSON file ---
    FILE_NAME = 'sentiment_vectors_with_prompts/sentiment_summaries_llama3_1b_NEWTS_test_100_articles_words_20250513_140435.json'
    input_scores_path = os.getenv('SCORES_PATH')
    scored_json_path = os.path.join(input_scores_path, FILE_NAME)
    output_plot_dir = os.path.join(os.getenv('PLOTS_PATH'), 'newts_summaries')
    base_plot_filename = FILE_NAME
    topic_id_to_plot = "tid1" # Specify which topic ID to plot (adjust if needed)
    reference_key_to_plot = "reference_text1" # Specify which reference key to plot (adjust if needed)

    # --- Load Data ---
    scored_data = load_scored_data(scored_json_path)

    if scored_data:
        # --- Generate Plots ---
        logger.info(f"--- Generating Plots for {base_plot_filename} ---")

        # 1. Sentiment Scores
        plot_sentiment_scores(
            scored_data=scored_data,
            output_dir=output_plot_dir,
            base_filename=base_plot_filename
        )

        # 2. Intrinsic Scores (with perplexity outlier removal)
        plot_intrinsic_scores(
            scored_data=scored_data,
            output_dir=output_plot_dir,
            base_filename=base_plot_filename,
            remove_perplexity_outliers=True,
            outlier_percentile=97.5
        )

        # 3. Topic Scores (for the specified topic ID)
        plot_topic_scores(
            scored_data=scored_data,
            topic_id=topic_id_to_plot,
            output_dir=output_plot_dir,
            base_filename=base_plot_filename
        )

        # 4. Extrinsic Scores (for the specified reference key)
        plot_extrinsic_scores(
            scored_data=scored_data,
            reference_key=reference_key_to_plot,
            output_dir=output_plot_dir,
            base_filename=base_plot_filename
        )

        # 5. Readability Scores
        plot_readability_scores(
            scored_data=scored_data,
            output_dir=output_plot_dir,
            base_filename=base_plot_filename
        )

        # 6. Toxicity Scores
        plot_toxicity_scores(
            scored_data=scored_data,
            output_dir=output_plot_dir,
            base_filename=base_plot_filename
        )

        logger.info("--- Plotting Complete ---")
    else:
        logger.error("Failed to load scored data. No plots generated.")

# --- Main Execution ---
if __name__ == '__main__':
    main()


