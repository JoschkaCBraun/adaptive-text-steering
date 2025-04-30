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

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

figsize = (6, 3) # Default figure size

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
        # Removed redundant warning check for experiment_information
        # if 'experiment_information' not in data:
        #     logger.warning("Loaded data missing 'experiment_information'. Plot titles might be generic.")
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
def prepare_plot_data(
    scored_summaries: Dict[str, Any],
    metrics_to_extract: Dict[str, List[str]],
    remove_perplexity_outliers: bool = False, # Default to False unless specifically for intrinsic plot
    outlier_percentile: float = 99.0
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, int], List[str]]:
    """
    Extracts specified metrics from scored_summaries, groups by strength,
    calculates counts, and handles nested extrinsic score averaging.

    Args:
        scored_summaries: The dictionary keyed by article_idx containing scores.
        metrics_to_extract: Dict defining which metrics to pull.
                           Keys are score categories (e.g., 'sentiment_scores'),
                           Values are lists of metric names (e.g., ['transformer', 'vader']).
                           For extrinsic scores, metric names like 'avg_rouge1' are expected.
        remove_perplexity_outliers: Whether to remove top percentile of perplexity scores.
                                    *Only* applied if True AND metric is 'perplexity'.
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

    # --- Perplexity Outlier Handling (only if requested) ---
    perplexity_threshold = None
    if remove_perplexity_outliers and 'intrinsic_scores' in metrics_to_extract and 'perplexity' in metrics_to_extract.get('intrinsic_scores', []):
        all_perplexity_values = []
        for article_idx, strength_dict in scored_summaries.items():
            if not isinstance(strength_dict, dict): continue
            for strength in sorted_strengths:
                score_dict = strength_dict.get(strength)
                if score_dict and isinstance(score_dict, dict):
                    intrinsic_scores = score_dict.get('intrinsic_scores')
                    if intrinsic_scores and isinstance(intrinsic_scores, dict):
                        perplexity = intrinsic_scores.get('perplexity')
                        if perplexity is not None and isinstance(perplexity, (int, float)) and math.isfinite(perplexity):
                            all_perplexity_values.append(float(perplexity))

        if all_perplexity_values:
            perplexity_threshold = np.percentile(all_perplexity_values, outlier_percentile)
            logger.info(f"Perplexity outlier threshold (top {100-outlier_percentile:.1f}%): {perplexity_threshold:.2f}")
    # --- End Perplexity Outlier Handling ---

    # --- Data Collection Loop ---
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
                            value = None # Reset value for each metric

                            # --- Special Handling for Averaged Extrinsic Scores ---
                            if score_category == 'extrinsic_scores' and metric.startswith('avg_'):
                                base_metric = metric[4:] # e.g., 'rouge1' from 'avg_rouge1'
                                ref1_scores = category_scores.get('reference_text1')
                                ref2_scores = category_scores.get('reference_text2')
                                values_to_avg = []
                                if ref1_scores and isinstance(ref1_scores, dict):
                                    v1 = ref1_scores.get(base_metric)
                                    if v1 is not None and isinstance(v1, (int, float)) and math.isfinite(v1):
                                        values_to_avg.append(float(v1))
                                if ref2_scores and isinstance(ref2_scores, dict):
                                    v2 = ref2_scores.get(base_metric)
                                    if v2 is not None and isinstance(v2, (int, float)) and math.isfinite(v2):
                                        values_to_avg.append(float(v2))

                                if values_to_avg: # Only calculate average if we have at least one valid score
                                    value = np.mean(values_to_avg)
                                else:
                                    logger.debug(f"Article {article_idx}, Strength {strength}: No valid values found for extrinsic metric '{base_metric}' in references.")
                                    value = None # Ensure value is None if no data
                            # --- End Extrinsic Handling ---
                            else:
                                # --- Standard Metric Extraction ---
                                value = category_scores.get(metric)

                            # --- Value Validation and Appending ---
                            if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                                current_value = float(value)
                                # Apply outlier filtering *only* for perplexity if enabled
                                if (remove_perplexity_outliers and
                                        score_category == 'intrinsic_scores' and
                                        metric == 'perplexity' and
                                        perplexity_threshold is not None and
                                        current_value > perplexity_threshold):
                                    logger.debug(f"Filtered out perplexity outlier: {current_value} > {perplexity_threshold}")
                                    continue # Skip adding this outlier

                                data_by_strength[strength][metric].append(current_value)
                            else:
                                # Log only if not an expected missing average
                                if not (score_category == 'extrinsic_scores' and metric.startswith('avg_')):
                                     logger.debug(f"Article {article_idx}, Strength {strength}, Metric {score_category}.{metric}: Invalid or missing value ({value}).")
                            # --- End Validation ---
                    else:
                       logger.debug(f"Article {article_idx}, Strength {strength}: Missing score category '{score_category}'.")
            else:
                 logger.debug(f"Article {article_idx}, Strength {strength}: Missing or invalid score dictionary.")
                 pass # No valid data for this strength/article

    # Use the count of valid score dictionaries for the title count
    counts_by_strength = dict(valid_score_dict_counts)

    # Log counts per strength for verification
    # for strength, count in counts_by_strength.items():
    #      logger.info(f"Strength {strength}: Found {count} valid score entries.")

    return dict(data_by_strength), counts_by_strength, sorted_strengths


# --- Plotting Function ---
def plot_scores_vs_strength(
    scored_data: Dict[str, Any],
    base_filename: str,
    output_dir: str,
    remove_perplexity_outliers: bool = True,
    outlier_percentile: float = 99.0
):
    """
    Generates and saves plots for various score categories vs. steering strength.
    Creates separate plots for Sentiment, Intrinsic, Extrinsic, Readability, and Toxicity.

    Args:
        scored_data: The dictionary containing experiment_info and scored_summaries.
        base_filename: Base name for the output plot files (e.g., 'plot_topic_30').
        output_dir: Directory to save the plots.
        remove_perplexity_outliers: Whether to remove top percentile of perplexity scores
                                    (only affects the Intrinsic plot).
        outlier_percentile: Percentile threshold for perplexity outlier removal.
    """
    if not scored_data or 'scored_summaries' not in scored_data:
        logger.error("Invalid or missing scored_data provided for plotting.")
        return

    scored_summaries = scored_data['scored_summaries']

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory '{output_dir}' ensured.")
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        return

    # ========================================================================
    # Plotting Helper Function (Reduces Repetition)
    # ========================================================================
    def _generate_plot(
        category_name: str,
        metrics_to_extract: Dict[str, List[str]],
        plot_filename_suffix: str,
        plot_title_prefix: str,
        primary_metrics: List[str], # Metrics for the primary Y-axis
        secondary_metrics: List[str], # Metrics for the secondary Y-axis (optional)
        metric_details: Dict[str, Dict[str, Any]], # color, label, scatter_marker, mean_marker
        primary_y_label: str,
        secondary_y_label: Optional[str] = None,
        primary_y_lim: Optional[Tuple[float, float]] = None,
        secondary_y_lim: Optional[Tuple[float, float]] = None,
        legend_ncol: int = 2,
        legend_loc: str = 'best',
        legend_bbox_anchor: Optional[Tuple[float, float]] = None,
        remove_outliers_flag: bool = False # Specific flag for this plot type
    ):
        """Internal helper to generate a single plot category."""
        logger.info(f"Preparing data for {category_name} plot...")
        plot_data, counts, sorted_strengths = prepare_plot_data(
            scored_summaries,
            metrics_to_extract,
            remove_perplexity_outliers=remove_outliers_flag, # Use specific flag
            outlier_percentile=outlier_percentile
        )

        if not plot_data or not any(plot_data.values()):
            logger.warning(f"No valid {category_name} data found to plot.")
            return

        logger.info(f"Generating {category_name} plot...")
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx() if secondary_metrics else None # Create secondary axis only if needed

        x_positions = np.arange(len(sorted_strengths))
        plot_handles = []
        plot_labels = []

        all_metrics_in_plot = primary_metrics + secondary_metrics

        for metric in all_metrics_in_plot:
            details = metric_details.get(metric)
            if not details:
                logger.warning(f"Missing plot details for metric '{metric}' in {category_name}. Skipping.")
                continue

            ax = ax2 if metric in secondary_metrics else ax1
            color = details['color']
            label = details['label']
            scatter_marker = details['scatter_marker']
            mean_marker = details['mean_marker']

            # Collect x and y for scatter and means
            all_x_scatter = []
            all_y_scatter = []
            y_means = []

            for i, strength in enumerate(sorted_strengths):
                y_values = plot_data.get(strength, {}).get(metric, [])
                if y_values:
                    # Jitter x positions slightly for scatter points to reduce overlap
                    jitter = np.random.normal(0, 0.05, len(y_values))
                    all_x_scatter.extend(x_positions[i] + jitter)
                    all_y_scatter.extend(y_values)
                    y_means.append(np.mean(y_values))
                else:
                    y_means.append(np.nan) # Use NaN if no data for mean line

            # Plot scatter points
            scatter_handle = ax.scatter(all_x_scatter, all_y_scatter, color=color, alpha=0.3, label=f"{label} Points", marker=scatter_marker, zorder=1, s=15) # Smaller points
            # Plot mean line (connect non-NaN points)
            valid_indices = ~np.isnan(y_means)
            if np.any(valid_indices):
                 mean_line_handle, = ax.plot(x_positions[valid_indices], np.array(y_means)[valid_indices], color='black', marker=mean_marker, linestyle='-', linewidth=1.5, markersize=5, label=f"Mean {label}", zorder=2)
                 plot_handles.extend([scatter_handle, mean_line_handle])
                 plot_labels.extend([f"{label} Points", f"Mean {label}"])
            else: # Only add scatter handle if no mean line can be drawn
                 plot_handles.append(scatter_handle)
                 plot_labels.append(f"{label} Points")


        # Configure axes
        ax1.set_xlabel("Steering Strength")
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(sorted_strengths)
        ax1.set_ylabel(primary_y_label, color='black' if not ax2 else metric_details[primary_metrics[0]]['color'])
        ax1.tick_params(axis='y', labelcolor='black' if not ax2 else metric_details[primary_metrics[0]]['color'])
        if primary_y_lim:
            ax1.set_ylim(primary_y_lim)

        if ax2 and secondary_metrics:
            ax2.set_ylabel(secondary_y_label, color=metric_details[secondary_metrics[0]]['color'])
            ax2.tick_params(axis='y', labelcolor=metric_details[secondary_metrics[0]]['color'])
            if secondary_y_lim:
                ax2.set_ylim(secondary_y_lim)
        elif ax2: # If ax2 exists but no secondary metrics (shouldn't happen with current logic, but safe)
             ax2.set_visible(False)


        # Average count for title
        avg_count = int(np.mean(list(counts.values()))) if counts else 0
        plot_title = f"{plot_title_prefix} vs. Steering Strength (Avg N={avg_count})"
        ax1.set_title(plot_title, fontsize=10)

        # Configure legend
        # Consolidate handles/labels: Create one entry per metric for the legend combining scatter and mean
        unique_labels_ordered = []
        handles_for_legend = []
        seen_labels = set()
        temp_handles = {} # Store one handle per base label

        # Get one representative handle for each metric (prefer mean line if available)
        for handle, full_label in zip(plot_handles, plot_labels):
             base_label = full_label.replace(" Points", "").replace("Mean ", "")
             if "Mean" in full_label: # Prioritize mean line handle
                 temp_handles[base_label] = handle
             elif base_label not in temp_handles: # Use scatter handle if mean wasn't found
                 temp_handles[base_label] = handle

        # Create final legend handles and labels in the desired order
        for metric in all_metrics_in_plot:
             details = metric_details.get(metric)
             if details and details['label'] in temp_handles:
                 base_label = details['label']
                 if base_label not in seen_labels:
                     handles_for_legend.append(temp_handles[base_label])
                     unique_labels_ordered.append(base_label)
                     seen_labels.add(base_label)


        fig.legend(handles_for_legend, unique_labels_ordered,
                   loc=legend_loc,
                   ncol=legend_ncol,
                   bbox_to_anchor=legend_bbox_anchor,
                   fontsize=8,
                   frameon=True)


        fig.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make space for legend if needed (esp. upper center)

        # Save plot
        plot_filename = f"{base_filename}_{plot_filename_suffix}.pdf"
        plot_filepath = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            logger.info(f"{category_name} plot saved to: {plot_filepath}")
        except Exception as e:
            logger.error(f"Failed to save {category_name} plot: {e}")
        plt.close(fig) # Close figure to free memory

    # ========================================================================
    # Define Plot Configurations & Generate Plots
    # ========================================================================

    # --- 1. Sentiment Plot ---
    _generate_plot(
        category_name="Sentiment",
        metrics_to_extract={'sentiment_scores': ['transformer', 'vader']},
        plot_filename_suffix="sentiment",
        plot_title_prefix="Sentiment Scores",
        primary_metrics=['transformer', 'vader'],
        secondary_metrics=[],
        metric_details={
            'transformer': {'color': 'tab:blue', 'label': 'Transformer', 'scatter_marker': 'o', 'mean_marker': 'o'},
            'vader': {'color': 'tab:red', 'label': 'VADER', 'scatter_marker': '^', 'mean_marker': '^'}
        },
        primary_y_label="Sentiment Score (-1 to +1)",
        primary_y_lim=(-1.05, 1.05),
        legend_ncol=2,
        legend_loc='lower center',
        legend_bbox_anchor=(0.5, -0.02), # Slightly below plot
        remove_outliers_flag=False
    )

    # --- 2. Intrinsic Quality Plot ---
    _generate_plot(
        category_name="Intrinsic",
        metrics_to_extract={'intrinsic_scores': ['perplexity', 'distinct_word_2', 'distinct_char_2']},
        plot_filename_suffix="intrinsic",
        plot_title_prefix="Intrinsic Quality",
        primary_metrics=['perplexity'],
        secondary_metrics=['distinct_word_2', 'distinct_char_2'],
        metric_details={
            'perplexity': {'color': 'tab:green', 'label': 'Perplexity', 'scatter_marker': 'o', 'mean_marker': 'o'},
            'distinct_word_2': {'color': 'tab:orange', 'label': 'Distinct-2 Words', 'scatter_marker': 's', 'mean_marker': 's'},
            'distinct_char_2': {'color': 'tab:purple', 'label': 'Distinct-2 Chars', 'scatter_marker': '^', 'mean_marker': '^'}
        },
        primary_y_label="Perplexity (Lower is Better)",
        secondary_y_label="Distinctness Score (0-1)",
        # primary_y_lim=(0, None), # Optional: set min perplexity if needed
        secondary_y_lim=(0, 1.05),
        legend_ncol=3,
        legend_loc='upper center',
        legend_bbox_anchor=(0.5, 1.15), # Above plot title
        remove_outliers_flag=remove_perplexity_outliers # Use the main flag here
    )

    # --- 3. Extrinsic Quality Plot ---
    _generate_plot(
        category_name="Extrinsic",
        # Request averaged metrics from prepare_plot_data
        metrics_to_extract={'extrinsic_scores': ['avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_bert_f1']},
        plot_filename_suffix="extrinsic",
        plot_title_prefix="Extrinsic Quality (vs Refs Avg)",
        primary_metrics=['avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_bert_f1'],
        secondary_metrics=[],
        metric_details={
            'avg_rouge1': {'color': 'tab:blue', 'label': 'ROUGE-1', 'scatter_marker': 'o', 'mean_marker': 'o'},
            'avg_rouge2': {'color': 'tab:green', 'label': 'ROUGE-2', 'scatter_marker': '^', 'mean_marker': '^'},
            'avg_rougeL': {'color': 'tab:red', 'label': 'ROUGE-L', 'scatter_marker': 's', 'mean_marker': 's'},
            'avg_bert_f1': {'color': 'tab:purple', 'label': 'BERT F1', 'scatter_marker': 'D', 'mean_marker': 'D'}
        },
        primary_y_label="Score (0-1, Higher is Better)",
        primary_y_lim=(0, 1.05),
        legend_ncol=2, # 4 items, 2 cols seems reasonable
        legend_loc='upper center',
        legend_bbox_anchor=(0.5, 1.15), # Above plot title
        remove_outliers_flag=False
    )

    # --- 4. Readability Plot ---
    _generate_plot(
        category_name="Readability",
        metrics_to_extract={'readability_scores': ['distilbert', 'deberta']},
        plot_filename_suffix="readability",
        plot_title_prefix="Readability Scores",
        primary_metrics=['distilbert'], # DistilBERT on left axis
        secondary_metrics=['deberta'],   # DeBERTa on right axis
        metric_details={
            'distilbert': {'color': 'tab:cyan', 'label': 'DistilBERT (Easier ↑)', 'scatter_marker': 'D', 'mean_marker': 'D'},
            'deberta': {'color': 'tab:olive', 'label': 'DeBERTa (Grade Lvl)', 'scatter_marker': 'P', 'mean_marker': 'P'}
        },
        primary_y_label="DistilBERT Score",
        secondary_y_label="DeBERTa Grade Level (Higher is Harder)",
        # primary_y_lim=(-3, 3), # Optional: Adjust based on typical DistilBERT range
        # secondary_y_lim=(0, 20), # Optional: Adjust DeBERTa grade level range
        legend_ncol=2,
        legend_loc='upper center',
        legend_bbox_anchor=(0.5, 1.15), # Above plot title
        remove_outliers_flag=False
    )

    # --- 5. Toxicity Plot ---
    _generate_plot(
        category_name="Toxicity",
        metrics_to_extract={'toxicity_scores': ['toxic_bert', 'severe_toxic_bert', 'roberta_toxicity']},
        plot_filename_suffix="toxicity",
        plot_title_prefix="Toxicity Scores",
        primary_metrics=['toxic_bert', 'severe_toxic_bert'], # These share left axis (0-1)
        secondary_metrics=['roberta_toxicity'],            # This gets right axis
        metric_details={
            'toxic_bert': {'color': 'tab:red', 'label': 'ToxicBERT (Toxic)', 'scatter_marker': 'o', 'mean_marker': 'o'},
            'severe_toxic_bert': {'color': 'tab:pink', 'label': 'ToxicBERT (Severe)', 'scatter_marker': '^', 'mean_marker': '^'},
            'roberta_toxicity': {'color': 'tab:brown', 'label': 'RoBERTa (Toxic)', 'scatter_marker': 's', 'mean_marker': 's'}
        },
        primary_y_label="ToxicBERT Score (0-1)",
        secondary_y_label="RoBERTa Score", # Scale might vary, don't fix to 0-1
        primary_y_lim=(0, 1.05),
        # secondary_y_lim=(0, 1.05), # Let matplotlib auto-scale RoBERTa axis unless needed
        legend_ncol=3,
        legend_loc='upper center',
        legend_bbox_anchor=(0.5, 1.15), # Above plot title
        remove_outliers_flag=False
    )

    logger.info("All plotting finished.")


# --- Example Usage ---
if __name__ == '__main__':
    # Configure logging for example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Path to your *scored* JSON file ---
    # Make sure this JSON contains sentiment, intrinsic, extrinsic, readability, AND toxicity scores
    scored_json_path = 'data/scores/topic_vectors/topic_summaries_llama3_1b_NEWTS_train_30_articles_topic_words_20250428_170346.json'
    # Output directory for plots
    output_directory = 'plots/topic_30_analysis' # Example output directory
    # Base filename for plots
    base_plot_filename = 'topic_30_llama3_1b_NEWTS' # Example base name

    # Load the scored data
    scored_data = load_scored_data(scored_json_path)

    # Check if data loaded successfully
    if scored_data:
        # Plot the scores vs. strength
        # Outlier removal for perplexity is True by default, but False for others.
        plot_scores_vs_strength(
            scored_data=scored_data,
            base_filename=base_plot_filename,
            output_dir=output_directory,
            remove_perplexity_outliers=True, # Keep perplexity outlier removal enabled
            outlier_percentile=99.0
        )
    else:
        logger.error("Failed to load data. Skipping plotting.")