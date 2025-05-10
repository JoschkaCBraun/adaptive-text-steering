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
        if 'scored_summaries' not in data or not isinstance(data['scored_summaries'], dict):
            logger.error("Loaded data missing 'scored_summaries' dictionary or it's not a dictionary.")
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
    data_by_strategy_key = defaultdict(lambda: defaultdict(list))
    counts_by_strategy_key = defaultdict(int)

    all_perplexity_values = []
    if remove_perplexity_outliers and perplexity_metric_path in metrics_to_extract:
        logger.info(f"Collecting '{perplexity_metric_path}' values for outlier detection across specified prompt strategies.")
        for article_idx, article_data in scored_summaries.items():
            if not isinstance(article_data, dict): continue
            # Look through all categories (topic, sentiment, etc.)
            for category, category_data in article_data.items():
                if not isinstance(category_data, dict): continue
                for strategy_key in ordered_prompt_strategy_keys:
                    score_dict = category_data.get(strategy_key)
                    if score_dict and isinstance(score_dict, dict):
                        perplexity_val = get_nested_value(score_dict, perplexity_metric_path)
                        if perplexity_val is not None and isinstance(perplexity_val, (int, float)) and math.isfinite(perplexity_val):
                            all_perplexity_values.append(float(perplexity_val))

    perplexity_threshold = None
    if all_perplexity_values and remove_perplexity_outliers:
        try:
            perplexity_threshold = np.percentile(all_perplexity_values, outlier_percentile)
            logger.info(f"'{perplexity_metric_path}' outlier threshold (top {100 - outlier_percentile:.1f}% of selected strategies): {perplexity_threshold:.2f}")
        except IndexError:
             logger.warning(f"Could not calculate percentile for {perplexity_metric_path}. Outlier removal skipped.")
             remove_perplexity_outliers = False

    valid_data_found = False
    for article_idx, article_data in scored_summaries.items():
        if not isinstance(article_data, dict):
            logger.debug(f"Article {article_idx}: Invalid or missing article_data.")
            continue

        # Look through all categories (topic, sentiment, etc.)
        for category, category_data in article_data.items():
            if not isinstance(category_data, dict): continue
            for strategy_key in ordered_prompt_strategy_keys:
                score_dict = category_data.get(strategy_key)
                if score_dict and isinstance(score_dict, dict):
                    counts_by_strategy_key[strategy_key] += 1
                    for metric_path in metrics_to_extract:
                        value = get_nested_value(score_dict, metric_path)
                        if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                            if (remove_perplexity_outliers and
                                metric_path == perplexity_metric_path and
                                perplexity_threshold is not None and
                                float(value) > perplexity_threshold):
                                logger.debug(f"Filtered out perplexity outlier for {strategy_key}: {value} > {perplexity_threshold}")
                                continue
                            data_by_strategy_key[strategy_key][metric_path].append(float(value))
                            valid_data_found = True
                        else:
                            logger.debug(f"Article {article_idx}, Strategy {strategy_key}, Metric {metric_path}: Invalid/missing value ({value}).")
                else:
                     logger.debug(f"Article {article_idx}, Strategy {strategy_key}: Missing or invalid score dictionary for this strategy.")
    
    if not valid_data_found:
        logger.warning("No valid data points found for any metric in the specified prompt strategies.")
        return None, None, ordered_prompt_strategy_keys

    for strategy_key in ordered_prompt_strategy_keys:
        logger.debug(f"Strategy Key '{strategy_key}': Processed {counts_by_strategy_key.get(strategy_key, 0)} articles.")

    return dict(data_by_strategy_key), dict(counts_by_strategy_key), ordered_prompt_strategy_keys



# --- Core Plotting Function ---

def _create_scatter_mean_plot(
    plot_data: Dict[str, Dict[str, List[float]]],
    ordered_strategy_keys: List[str], 
    x_axis_display_labels: List[str], 
    plot_config: Dict[str, Any],
    title: str,
    xlabel: str,
    output_filepath: str,
    figsize: Tuple[float, float] = FIGSIZE
):
    logger.info(f"Generating plot: {title} -> {output_filepath}")
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = None
    axes = {'primary': ax1}

    if plot_config.get('secondary_axis'):
        ax2 = ax1.twinx()
        axes['secondary'] = ax2

    x_positions = np.arange(len(ordered_strategy_keys))
    all_handles_labels = {} 

    for axis_key, ax in axes.items():
        axis_config = plot_config.get(f"{axis_key}_axis")
        if not axis_config or not axis_config.get('metrics'): continue

        for metric_info in axis_config['metrics']:
            metric_path = metric_info['name']
            metric_label_base = metric_info['label'] # E.g., "Transformer Sent."
            metric_color = metric_info['color']
            scatter_marker = metric_info.get('scatter_marker', 'o')
            mean_marker = metric_info.get('mean_marker', 'o')

            all_x_scatter_points = []
            all_y_scatter_points = []
            y_means = []

            for i, strategy_key in enumerate(ordered_strategy_keys):
                y_values = plot_data.get(strategy_key, {}).get(metric_path, [])
                if y_values:
                    jitter = np.random.normal(0, 0.04, len(y_values)) # Reduced jitter
                    all_x_scatter_points.extend(x_positions[i] + jitter)
                    all_y_scatter_points.extend(y_values)
                    y_means.append(np.mean(y_values))
                else:
                    y_means.append(np.nan)

            scatter_legend_label = f"{metric_label_base}" # Simpler legend: "Transformer Sent."
            mean_legend_label = f"{metric_label_base} (Mean)"

            if all_x_scatter_points:
                scatter_handle = ax.scatter(
                    all_x_scatter_points, all_y_scatter_points,
                    color=metric_color, alpha=plot_config.get('scatter_alpha', 0.5), # Slightly more transparent
                    label=scatter_legend_label, marker=scatter_marker, zorder=2 # zorder for scatter
                )
                if scatter_legend_label not in all_handles_labels:
                    all_handles_labels[scatter_legend_label] = scatter_handle

            valid_indices = ~np.isnan(y_means)
            if np.any(valid_indices):
                mean_line_handle, = ax.plot(
                    x_positions[valid_indices], np.array(y_means)[valid_indices],
                    color=plot_config.get('mean_line_color', metric_color), # Default to metric color
                    marker=mean_marker,
                    linestyle=plot_config.get('mean_line_style', '-'),
                    label=mean_legend_label, zorder=10 # Mean line on top
                )
                if mean_legend_label not in all_handles_labels:
                     all_handles_labels[mean_legend_label] = mean_line_handle

        ax.set_ylabel(axis_config['ylabel'])
        if axis_config.get('ylim'): ax.set_ylim(axis_config['ylim'])
        ax.tick_params(axis='y')
        
        # Ensure y-axis ticks are reasonable for 0-1 scales
        if ax.get_ylim()[1] <= 1.1 and ax.get_ylim()[0] >= -0.1 and not axis_config.get('ylim') == minus_one_to_one_ylim_with_padding:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))


    ax1.set_xlabel(xlabel, labelpad=10)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_axis_display_labels)
    fig.suptitle(title, fontsize=12, y=0.98) # Use suptitle for main title
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    unique_labels = list(all_handles_labels.keys())
    unique_handles = [all_handles_labels[label] for label in unique_labels]
    
    legend_opts = plot_config.get('legend_opts', {})
    default_legend_opts = {
        'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.12), 
        'ncol': min(2, len(unique_handles)), 'frameon': True, 'edgecolor': 'black',
        'fancybox': True, 'fontsize': 8, 
    }
    final_legend_opts = {**default_legend_opts, **legend_opts}

    if unique_handles:
        ax1.legend(unique_handles, unique_labels, **final_legend_opts)
    
    fig.tight_layout(rect=[0, 0.1, 1, 0.93]) # Adjust rect for suptitle and legend

    output_dir_path = os.path.dirname(output_filepath)
    if output_dir_path: os.makedirs(output_dir_path, exist_ok=True)
    try:
        plt.savefig(output_filepath, dpi=300) # bbox_inches='tight' can conflict with suptitle/legend placement
        logger.info(f"Plot saved to: {output_filepath}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_filepath}: {e}")
    finally:
        plt.close(fig)

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


def plot_sentiment_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str):
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})
    main_plot_label = exp_info.get('main_plot_label', base_filename)

    strategy_map = {
        "Discouraged": "sentiment_negative_encouraged",
        "Neutral": "neutral",
        "Encouraged": "sentiment_positive_encouraged"
    }
    ordered_keys, display_labels = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, ["Negative", "Neutral", "Positive"])
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")): # Check a sample article
         logger.warning(f"Not all required strategy keys for sentiment plot found in data for {main_plot_label}. Skipping."); return

    metric_paths = ['sentiment_scores.transformer', 'sentiment_scores.vader']
    plot_config = {
        'primary_axis': {
            'metrics': [
                {'name': 'sentiment_scores.transformer', 'label': 'Transformer', 'color': 'tab:blue', 'mean_marker': 'o', 'scatter_marker': 'o'},
                {'name': 'sentiment_scores.vader', 'label': 'VADER', 'color': 'tab:red', 'mean_marker': '^', 'scatter_marker': '^'}
            ], 'ylabel': 'Sentiment Score', 'ylim': minus_one_to_one_ylim_with_padding
        }, 'legend_opts': {'ncol': 2}
    }
    plot_data, _, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for sentiment plot {main_plot_label}. Skipping."); return
    _create_scatter_mean_plot(plot_data, ordered_keys, display_labels, plot_config,
        title=f"Sentiment Scores vs. Prompt Strategy\n({main_plot_label})",
        xlabel="Prompted Sentiment",
        output_filepath=os.path.join(output_dir, f"{base_filename}_sentiment.pdf"))


def plot_intrinsic_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str, remove_perplexity_outliers: bool = True, outlier_percentile: float = 97.5):
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})
    main_plot_label = exp_info.get('main_plot_label', base_filename)

    strategy_map = { # Using sentiment prompts as proxy for intrinsic quality changes
        "Discouraged": "sentiment_negative_encouraged", "Neutral": "neutral", "Encouraged": "sentiment_positive_encouraged"
    }
    ordered_keys, display_labels = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, ["Sent-Neg", "Neutral", "Sent-Pos"])
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for intrinsic plot found for {main_plot_label}. Skipping."); return

    metric_paths = ['intrinsic_scores.perplexity', 'intrinsic_scores.distinct_word_2', 'intrinsic_scores.distinct_char_2']
    plot_config = {
        'primary_axis': {'metrics': [{'name': 'intrinsic_scores.perplexity', 'label': 'Perplexity', 'color': 'tab:green'}],'ylabel': 'Perplexity', 'ylim': None},
        'secondary_axis': {'metrics': [
                 {'name': 'intrinsic_scores.distinct_word_2', 'label': 'Distinct-2 Words', 'color': 'tab:orange', 'mean_marker': 's', 'scatter_marker': 's'},
                 {'name': 'intrinsic_scores.distinct_char_2', 'label': 'Distinct-2 Chars', 'color': 'tab:purple', 'mean_marker': '^', 'scatter_marker': '^'}
             ],'ylabel': 'Distinctness Score', 'ylim': (-0.05, 1.25)},
        'legend_opts': {'ncol': 3}
    }
    plot_data, _, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys, 'intrinsic_scores.perplexity', remove_perplexity_outliers, outlier_percentile)
    if not plot_data: logger.warning(f"No data for intrinsic plot {main_plot_label}. Skipping."); return
    _create_scatter_mean_plot(plot_data, ordered_keys, display_labels, plot_config,
        title=f"Intrinsic Quality vs. Prompt Strategy\n({main_plot_label})",
        xlabel="Prompt Strategy (based on Sentiment Prompts)",
        output_filepath=os.path.join(output_dir, f"{base_filename}_intrinsic.pdf"))


def plot_topic_scores(scored_data: Dict[str, Any], topic_id: str, output_dir: str, base_filename: str):
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})
    main_plot_label = exp_info.get('main_plot_label', base_filename)
    alt_topic_id = "tid2" if topic_id == "tid1" else "tid1" # Simple toggle

    strategy_map = {
        "Discouraged": f"topic_{alt_topic_id}_encouraged", "Neutral": "neutral", "Encouraged": f"topic_{topic_id}_encouraged"
    }
    custom_labels = [f"Focus {alt_topic_id.upper()}", "Neutral", f"Focus {topic_id.upper()}"]
    ordered_keys, display_labels = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, custom_labels)
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for topic {topic_id} plot found for {main_plot_label}. Skipping."); return

    metric_paths = [f'topic_scores.{topic_id}.dict', f'topic_scores.{topic_id}.tokenize', f'topic_scores.{topic_id}.lemmatize']
    plot_config = {
        'primary_axis': {'metrics': [
                {'name': f'topic_scores.{topic_id}.dict', 'label': f'{topic_id.upper()} Dict', 'color': 'tab:cyan'},
                {'name': f'topic_scores.{topic_id}.tokenize', 'label': f'{topic_id.upper()} Token', 'color': 'tab:olive', 'mean_marker': 's', 'scatter_marker': 's'},
                {'name': f'topic_scores.{topic_id}.lemmatize', 'label': f'{topic_id.upper()} Lemma', 'color': 'tab:pink', 'mean_marker': '^', 'scatter_marker': '^'}
            ],'ylabel': f'Topic Score ({topic_id.upper()})', 'ylim': zero_to_one_ylim_with_padding},
        'legend_opts': {'ncol': 3}
    }
    plot_data, _, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for topic {topic_id} plot {main_plot_label}. Skipping."); return
    _create_scatter_mean_plot(plot_data, ordered_keys, display_labels, plot_config,
        title=f"Topic Adherence ({topic_id.upper()}) vs. Prompt Strategy\n({main_plot_label})",
        xlabel=f"Prompt Strategy (Targeting {topic_id.upper()})",
        output_filepath=os.path.join(output_dir, f"{base_filename}_topic_{topic_id}.pdf"))


def plot_extrinsic_scores(scored_data: Dict[str, Any], reference_key: str, output_dir: str, base_filename: str):
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})
    main_plot_label = exp_info.get('main_plot_label', base_filename)
    strategy_map = { # Using sentiment prompts as proxy
        "Discouraged": "sentiment_negative_encouraged", "Neutral": "neutral", "Encouraged": "sentiment_positive_encouraged"
    }
    ordered_keys, display_labels = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, ["Sent-Neg", "Neutral", "Sent-Pos"])
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for extrinsic plot found for {main_plot_label}. Skipping."); return

    metric_paths = [f'extrinsic_scores.{reference_key}.rouge1', f'extrinsic_scores.{reference_key}.rouge2', f'extrinsic_scores.{reference_key}.rougeL', f'extrinsic_scores.{reference_key}.bert_f1']
    plot_config = {
        'primary_axis': {'metrics': [
                {'name': f'extrinsic_scores.{reference_key}.rouge1', 'label': 'ROUGE-1', 'color': 'tab:blue'},
                {'name': f'extrinsic_scores.{reference_key}.rouge2', 'label': 'ROUGE-2', 'color': 'tab:red', 'mean_marker': 's', 'scatter_marker': 's'},
                {'name': f'extrinsic_scores.{reference_key}.rougeL', 'label': 'ROUGE-L', 'color': 'tab:green', 'mean_marker': '^', 'scatter_marker': '^'}
            ],'ylabel': 'ROUGE Score', 'ylim': zero_to_one_ylim_with_padding},
        'secondary_axis': {'metrics': [{'name': f'extrinsic_scores.{reference_key}.bert_f1', 'label': 'BERT F1', 'color': 'tab:purple', 'mean_marker': 'd', 'scatter_marker': 'd'}],
             'ylabel': 'BERTScore F1', 'ylim': zero_to_one_ylim_with_padding},
        'legend_opts': {'ncol': 2}
    }
    plot_data, _, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for extrinsic plot {main_plot_label} (ref: {reference_key}). Skipping."); return
    _create_scatter_mean_plot(plot_data, ordered_keys, display_labels, plot_config,
        title=f"Extrinsic Quality ({reference_key}) vs. Prompt Strategy\n({main_plot_label})",
        xlabel="Prompt Strategy (based on Sentiment Prompts)",
        output_filepath=os.path.join(output_dir, f"{base_filename}_extrinsic_{reference_key}.pdf"))


def plot_readability_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str):
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})
    main_plot_label = exp_info.get('main_plot_label', base_filename)
    strategy_map = {
        "Discouraged": "readability_simple_encouraged", "Neutral": "neutral", "Encouraged": "readability_complex_encouraged"
    }
    ordered_keys, display_labels = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, ["Simple", "Neutral", "Complex"])
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for readability plot found for {main_plot_label}. Skipping."); return

    metric_paths = ['readability_scores.distilbert', 'readability_scores.deberta']
    plot_config = { # DeBERTa: higher is simpler/more readable. DistilBERT: lower is simpler/more readable.
        'primary_axis': {'metrics': [{'name': 'readability_scores.deberta', 'label': 'DeBERTa', 'color': 'tab:blue'}],'ylabel': 'DeBERTa (Higher is Simpler)', 'ylim': None},
        'secondary_axis': {'metrics': [{'name': 'readability_scores.distilbert', 'label': 'DistilBERT', 'color': 'tab:red', 'mean_marker': '^', 'scatter_marker': '^'}],'ylabel': 'DistilBERT (Lower is Simpler)', 'ylim': None},
        'legend_opts': {'ncol': 2}
    }
    plot_data, _, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for readability plot {main_plot_label}. Skipping."); return
    _create_scatter_mean_plot(plot_data, ordered_keys, display_labels, plot_config,
        title=f"Readability Scores vs. Prompt Strategy\n({main_plot_label})",
        xlabel="Prompted Readability Level",
        output_filepath=os.path.join(output_dir, f"{base_filename}_readability.pdf"))


def plot_toxicity_scores(scored_data: Dict[str, Any], output_dir: str, base_filename: str):
    if 'scored_summaries' not in scored_data: return
    exp_info = scored_data.get('experiment_information', {})
    main_plot_label = exp_info.get('main_plot_label', base_filename)
    strategy_map = {
        "Discouraged": "toxicity_avoided", "Neutral": "neutral", "Encouraged": "toxicity_encouraged"
    }
    ordered_keys, display_labels = _get_ordered_keys_and_labels(strategy_map, CONCEPTUAL_CATEGORIES_ORDER, ["Avoided", "Neutral", "Encouraged"])
    if not all(key in scored_data['scored_summaries'].get(next(iter(scored_data['scored_summaries'])), {}) for key in ordered_keys if not key.startswith("MISSING")):
         logger.warning(f"Not all required strategy keys for toxicity plot found for {main_plot_label}. Skipping."); return

    metric_paths = ['toxicity_scores.toxic_bert', 'toxicity_scores.severe_toxic_bert', 'toxicity_scores.roberta_toxicity']
    plot_config = {
        'primary_axis': {'metrics': [
                {'name': 'toxicity_scores.toxic_bert', 'label': 'Toxic BERT', 'color': 'tab:blue'},
                {'name': 'toxicity_scores.severe_toxic_bert', 'label': 'Severe Toxic BERT', 'color': 'tab:red', 'mean_marker': 's', 'scatter_marker': 's'},
                {'name': 'toxicity_scores.roberta_toxicity', 'label': 'RoBERTa Toxicity', 'color': 'tab:green', 'mean_marker': '^', 'scatter_marker': '^'}
            ],'ylabel': 'Toxicity Score', 'ylim': zero_to_one_ylim_with_padding},
        'legend_opts': {'ncol': 3}
    }
    plot_data, _, _ = prepare_plot_data(scored_data['scored_summaries'], metric_paths, ordered_keys)
    if not plot_data: logger.warning(f"No data for toxicity plot {main_plot_label}. Skipping."); return
    _create_scatter_mean_plot(plot_data, ordered_keys, display_labels, plot_config,
        title=f"Toxicity Scores vs. Prompt Strategy\n({main_plot_label})",
        xlabel="Prompted Toxicity Level",
        output_filepath=os.path.join(output_dir, f"{base_filename}_toxicity.pdf"))


def main() -> None:
    """
    This function generates plots for the prompt engineering experiments on the NEWTS dataset.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    FILE_NAME = "prompt_engineering/prompt_engineering_summaries_llama3_1b_NEWTS_train_3_articles_20250509_160915.json" # Replace with your actual file
    
    scores_path = os.getenv('SCORES_PATH') 
    plots_path = os.getenv('PLOTS_PATH')
    
    scored_json_path = os.path.join(scores_path, FILE_NAME)
    output_plot_dir = os.path.join(plots_path, "prompt_engineering")
    base_plot_filename_prefix = FILE_NAME # For plot filenames

    logger.info(f"Input scored JSON: {scored_json_path}")
    logger.info(f"Output plot directory: {output_plot_dir}")

    scored_data = load_scored_data(scored_json_path)

    if scored_data:
        # Store a label for plots from experiment_information or fallback to filename
        exp_name = scored_data.get('experiment_information', {}).get('experiment_name', 'PE Exp')
        model_alias = scored_data.get('experiment_information', {}).get('model_alias', '')
        num_articles = scored_data.get('experiment_information', {}).get('num_articles', 'N/A')
        main_plot_label = f"{exp_name} ({model_alias}, {num_articles} art.)"
        if 'experiment_information' in scored_data: # Pass it down for consistent plot titles
            scored_data['experiment_information']['main_plot_label'] = main_plot_label
        else: # Should not happen due to load_scored_data, but defensive
            scored_data['experiment_information'] = {'main_plot_label': base_plot_filename_prefix}


        os.makedirs(output_plot_dir, exist_ok=True)
        logger.info(f"--- Generating Plots for: {main_plot_label} ---")

        plot_sentiment_scores(scored_data, output_plot_dir, base_plot_filename_prefix)
        plot_intrinsic_scores(scored_data, output_plot_dir, base_plot_filename_prefix)
        
        # Assuming 'tid1' and 'tid2' are the topic IDs used in the experiment.
        # Adapt if your topic IDs are different or discovered dynamically.
        plot_topic_scores(scored_data, "tid1", output_plot_dir, base_plot_filename_prefix)
        plot_topic_scores(scored_data, "tid2", output_plot_dir, base_plot_filename_prefix)
        
        # Assuming 'reference_text1' is a key for extrinsic scores.
        plot_extrinsic_scores(scored_data, "reference_text1", output_plot_dir, base_plot_filename_prefix)
        
        plot_readability_scores(scored_data, output_plot_dir, base_plot_filename_prefix)
        plot_toxicity_scores(scored_data, output_plot_dir, base_plot_filename_prefix)

        logger.info(f"--- Plotting Complete for: {main_plot_label} ---")
    else:
        logger.error(f"Failed to load scored data from {scored_json_path}. No plots generated.")

# --- Main Execution ---
if __name__ == '__main__':
    main()
