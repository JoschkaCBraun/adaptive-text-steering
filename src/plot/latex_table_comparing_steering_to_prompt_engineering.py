# Standard library imports
import os
import math
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Literal

# Third-party imports
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions (adapted from provided scripts) ---

def get_nested_value(data_dict: Dict, key_path: str, default: Any = None) -> Any:
    """Safely retrieves a value from a nested dictionary using a dot-separated path."""
    keys = key_path.split('.')
    value = data_dict
    try:
        for key_item in keys:
            if isinstance(value, dict):
                value = value.get(key_item)
            elif isinstance(value, list) and key_item.isdigit():
                try:
                    value = value[int(key_item)]
                except IndexError:
                    logger.debug(f"Index {key_item} out of bounds in path '{key_path}'.")
                    return default
            else:
                logger.debug(f"Cannot traverse further at key '{key_item}' in path '{key_path}'. Value is not a dict/list: {value}")
                return default
            if value is None:
                 logger.debug(f"Key '{key_item}' not found in path '{key_path}'.")
                 return default
        return value
    except Exception as e:
        logger.debug(f"Error accessing nested key '{key_path}': {e}")
        return default

def load_scored_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads the scored summary data from a JSON file."""
    logger.info(f"Attempting to load scored data from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Scored data loaded successfully from {file_path}.")
        if 'scored_summaries' not in data or not isinstance(data['scored_summaries'], dict):
            logger.error(f"Loaded data from {file_path} missing 'scored_summaries' dictionary or it's not a dict.")
            return None
        if 'experiment_information' not in data:
             logger.warning(f"Loaded data from {file_path} missing 'experiment_information'.")
             data['experiment_information'] = {}
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {file_path} - {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading data from {file_path}: {e}", exc_info=True)
        return None

def calculate_mean_variance_from_scores(scores: List[float], mean_precision: int = 2, var_precision: int = 1) -> str:
    """Calculates mean and variance from a list of scores and formats them."""
    if not scores:
        return "N/A"
    try:
        mean_val = np.mean(scores)
        var_val = np.var(scores)
        if var_val < 0: # Handle potential float precision issues for very small variances
            var_val = 0.0
        
        var_str = f"{var_val:.{var_precision}f}"
        if var_str == f"-0.{'0'*var_precision}": 
            var_str = f"0.{'0'*var_precision}"

        return f"{mean_val:.{mean_precision}f} $\\pm$ {var_str}"
    except Exception as e:
        logger.error(f"Error calculating mean/variance from scores: {e}")
        return "Error"

def format_calculated_stats(mean_val: float, var_val: float, mean_precision: int = 2, var_precision: int = 1) -> str:
    """Formats pre-calculated mean and variance."""
    if var_val < 0: var_val = 0.0 # Should be handled before calling, but as a safeguard
    var_str = f"{var_val:.{var_precision}f}"
    if var_str == f"-0.{'0'*var_precision}":
        var_str = f"0.{'0'*var_precision}"
    return f"{mean_val:.{mean_precision}f} $\\pm$ {var_str}"


def get_values_for_condition(
    scored_summaries: Dict[str, Dict[str, Any]],
    condition_key: str, 
    metric_json_path: str 
) -> List[float]:
    """
    Extracts all valid numerical values for a specific metric under a given
    experimental condition (e.g., a specific prompt strategy or steering strength) across all articles.
    """
    collected_values = []
    if not scored_summaries:
        logger.warning(f"Scored summaries is empty or None when trying to get values for condition_key '{condition_key}' and metric_json_path '{metric_json_path}'")
        return []

    for article_idx, article_data in scored_summaries.items():
        if not isinstance(article_data, dict):
            logger.debug(f"Article {article_idx} data is not a dict, skipping.")
            continue

        condition_specific_data_block = article_data.get(condition_key)
        if condition_specific_data_block is None:
            logger.debug(f"Condition key '{condition_key}' not found for article {article_idx}.")
            continue
        if not isinstance(condition_specific_data_block, dict):
            logger.debug(f"Data for condition '{condition_key}' in article {article_idx} is not a dict, skipping.")
            continue

        value = get_nested_value(condition_specific_data_block, metric_json_path)

        if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
            collected_values.append(float(value))
        else:
            logger.debug(f"Article {article_idx}, ConditionKey {condition_key}, MetricPath {metric_json_path}: Invalid or missing value ({value}).")

    if not collected_values:
        logger.warning(f"No valid values collected for condition_key '{condition_key}', metric_json_path '{metric_json_path}'.")
    return collected_values

# --- Configuration for Table Generation ---

TABLE_GENERATOR_CONFIG = {
    "Topic": {
        "sub_metrics": [
            {"name": "dict", "path": "topic_scores.tid1.dict"},
            {"name": "stem", "path": "topic_scores.tid1.stem"},
            {"name": "lemmatize", "path": "topic_scores.tid1.lemmatize"},
            {"name": "tokenize", "path": "topic_scores.tid1.tokenize"}
        ],
        "prompt_key_encourage": "topic_tid1_encouraged",
        "prompt_key_discourage": "topic_tid2_encouraged", 
        "prompt_key_neutral": "neutral",
        "steering_strength_keys": {"-2": "-2", "-1": "-1", "1": "1", "2": "2"} 
    },
    "Sentiment": {
        "sub_metrics": [
            {"name": "VADER", "path": "sentiment_scores.vader"},
            {"name": "Transformer", "path": "sentiment_scores.transformer"}
        ],
        "prompt_key_encourage": "sentiment_positive_encouraged",
        "prompt_key_discourage": "sentiment_negative_encouraged",
        "prompt_key_neutral": "neutral",
        "steering_strength_keys": {"-2": "-2", "-1": "-1", "1": "1", "2": "2"}
    },
    "Readability": {
        "sub_metrics": [
            {"name": "DistilBERT", "path": "readability_scores.distilbert"},
            {"name": "DeBERTa", "path": "readability_scores.deberta"}
        ],
        "prompt_key_encourage": "readability_simple_encouraged",
        "prompt_key_discourage": "readability_complex_encouraged",
        "prompt_key_neutral": "neutral",
        "steering_strength_keys": {"-2": "-2", "-1": "-1", "1": "1", "2": "2"}
    },
    "Toxic": {
        "sub_metrics": [
            {"name": "ToxicBERT", "path": "toxicity_scores.toxic_bert"},
            {"name": "Severe Toxic", "path": "toxicity_scores.severe_toxic_bert"},
            {"name": "RoBERTa", "path": "toxicity_scores.roberta_toxicity"}
        ],
        "prompt_key_encourage": "toxicity_encouraged", 
        "prompt_key_discourage": "toxicity_avoided",    
        "prompt_key_neutral": "neutral",
        "steering_strength_keys": {"-2": "-2", "-1": "-1", "1": "1", "2": "2"}
    }
}

TABLE_COLUMNS_CONFIG = [
    ("steering", "-2"),
    ("steering", "-1"),
    ("prompting", "Discourage"),
    ("prompting", "Neutral"),
    ("prompting", "Encourage"),
    ("steering", "1"),
    ("steering", "2")
]

BEHAVIOR_ORDER = ["Topic", "Sentiment", "Readability", "Toxic"]


def generate_single_table_latex(
    table_type: Literal["detailed", "averaged"],
    prompt_eng_summaries: Dict[str, Dict[str, Any]],
    loaded_steering_summaries: Dict[str, Optional[Dict[str, Dict[str, Any]]]]
) -> str:
    latex_table_rows = []
    header = r"""\begin{tabular}{@{}l*{7}{c}@{}}
\toprule
& \multicolumn{2}{c}{Steering with strength $\lambda$} & \multicolumn{3}{c}{Prompting model for behavior} & \multicolumn{2}{c}{Steering with strength $\lambda$} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8}
Behavior & $\lambda = -2$ & $\lambda = -1$ & Discourage & Neutral & Encourage & $\lambda = 1$ & $\lambda = 2$ \\
\midrule"""
    latex_table_rows.append(header)

    mean_precision = 2
    var_precision = 1

    for behavior_key in BEHAVIOR_ORDER:
        logger.info(f"Processing behavior: {behavior_key} for {table_type} table.")
        behavior_config = TABLE_GENERATOR_CONFIG[behavior_key]
        
        if table_type == "detailed":
            latex_table_rows.append(f"{behavior_key} & & & & & & & \\\\") # Main behavior category row

        all_sub_metrics_for_behavior = behavior_config["sub_metrics"]

        if table_type == "averaged":
            row_cells = [behavior_key] 
            for col_type, col_condition_concept in TABLE_COLUMNS_CONFIG:
                list_of_individual_means = []
                list_of_individual_variances = []

                for sub_metric_conf in all_sub_metrics_for_behavior:
                    metric_json_path = sub_metric_conf["path"]
                    current_raw_scores = []

                    if col_type == "steering":
                        steering_summaries = loaded_steering_summaries.get(behavior_key)
                        condition_key_in_json = behavior_config["steering_strength_keys"].get(str(col_condition_concept))
                        if steering_summaries and condition_key_in_json:
                            current_raw_scores = get_values_for_condition(steering_summaries, condition_key_in_json, metric_json_path)
                    elif col_type == "prompting":
                        condition_key_in_json = None
                        if col_condition_concept == "Encourage": condition_key_in_json = behavior_config["prompt_key_encourage"]
                        elif col_condition_concept == "Discourage": condition_key_in_json = behavior_config["prompt_key_discourage"]
                        elif col_condition_concept == "Neutral": condition_key_in_json = behavior_config["prompt_key_neutral"]
                        
                        if prompt_eng_summaries and condition_key_in_json:
                            current_raw_scores = get_values_for_condition(prompt_eng_summaries, condition_key_in_json, metric_json_path)
                    
                    if current_raw_scores:
                        individual_mean = np.mean(current_raw_scores)
                        individual_variance = np.var(current_raw_scores)
                        if individual_variance < 0: individual_variance = 0.0
                        
                        list_of_individual_means.append(individual_mean)
                        list_of_individual_variances.append(individual_variance)
                
                if list_of_individual_means: # Check if any sub-metric had data
                    averaged_mean = np.mean(list_of_individual_means)
                    averaged_variance = np.mean(list_of_individual_variances)
                    cell_value_str = format_calculated_stats(averaged_mean, averaged_variance, mean_precision, var_precision)
                else:
                    cell_value_str = "N/A"
                row_cells.append(cell_value_str)
            latex_table_rows.append(" & ".join(row_cells) + r" \\")

        elif table_type == "detailed":
            for sub_metric_conf in all_sub_metrics_for_behavior:
                sub_metric_name = sub_metric_conf["name"]
                metric_json_path = sub_metric_conf["path"]
                row_cells = [f"  {sub_metric_name}"]

                for col_type, col_condition_concept in TABLE_COLUMNS_CONFIG:
                    metric_values = []
                    if col_type == "steering":
                        steering_summaries = loaded_steering_summaries.get(behavior_key)
                        condition_key_in_json = behavior_config["steering_strength_keys"].get(str(col_condition_concept))
                        if steering_summaries and condition_key_in_json :
                            metric_values = get_values_for_condition(steering_summaries, condition_key_in_json, metric_json_path)
                    elif col_type == "prompting":
                        condition_key_in_json = None
                        if col_condition_concept == "Encourage": condition_key_in_json = behavior_config["prompt_key_encourage"]
                        elif col_condition_concept == "Discourage": condition_key_in_json = behavior_config["prompt_key_discourage"]
                        elif col_condition_concept == "Neutral": condition_key_in_json = behavior_config["prompt_key_neutral"]
                        
                        if prompt_eng_summaries and condition_key_in_json:
                            metric_values = get_values_for_condition(prompt_eng_summaries, condition_key_in_json, metric_json_path)
                    
                    row_cells.append(calculate_mean_variance_from_scores(metric_values, mean_precision, var_precision))
                latex_table_rows.append(" & ".join(row_cells) + r" \\")
        
        if table_type == "detailed" and behavior_key != BEHAVIOR_ORDER[-1]:
             latex_table_rows.append(r"\midrule") # Horizontal line between behavior groups in detailed table

    latex_table_rows.append(r"\bottomrule")
    latex_table_rows.append(r"\end{tabular}")
    return "\n".join(latex_table_rows)


# --- Main Function ---
def generate_all_latex_tables(
    prompt_eng_file_path: str,
    topic_steering_file_path: str,
    sentiment_steering_file_path: str,
    readability_steering_file_path: str,
    toxicity_steering_file_path: str,
    output_dir: str
) -> None:
    logger.info("Starting LaTeX table generation for detailed and averaged metrics.")

    prompt_eng_data = load_scored_data(prompt_eng_file_path)
    if not prompt_eng_data:
        logger.error(f"CRITICAL: Failed to load prompt engineering data from {prompt_eng_file_path}. Aborting.")
        return
    prompt_eng_summaries = prompt_eng_data['scored_summaries']

    steering_files_map = {
        "Topic": topic_steering_file_path,
        "Sentiment": sentiment_steering_file_path,
        "Readability": readability_steering_file_path,
        "Toxic": toxicity_steering_file_path,
    }

    loaded_steering_summaries = {}
    for behavior_type, file_path in steering_files_map.items():
        data = load_scored_data(file_path)
        if data and 'scored_summaries' in data:
            loaded_steering_summaries[behavior_type] = data['scored_summaries']
        else:
            logger.warning(f"Could not load steering data for {behavior_type} from {file_path} or 'scored_summaries' missing.")
            loaded_steering_summaries[behavior_type] = None 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    logger.info("Generating Detailed Metrics Table...")
    detailed_latex_string = generate_single_table_latex("detailed", prompt_eng_summaries, loaded_steering_summaries)
    detailed_output_file = os.path.join(output_dir, "steering_vs_prompting_detailed_metrics.tex")
    try:
        with open(detailed_output_file, 'w', encoding='utf-8') as f: 
            f.write(detailed_latex_string)
        logger.info(f"Detailed LaTeX table successfully written to {detailed_output_file}")
    except IOError as e:
        logger.error(f"Failed to write detailed LaTeX table to {detailed_output_file}: {e}")

    logger.info("Generating Averaged Metrics Table (new method)...")
    averaged_latex_string = generate_single_table_latex("averaged", prompt_eng_summaries, loaded_steering_summaries)
    averaged_output_file = os.path.join(output_dir, "steering_vs_prompting_averaged_metrics.tex") # New filename for new method
    try:
        with open(averaged_output_file, 'w', encoding='utf-8') as f:
            f.write(averaged_latex_string)
        logger.info(f"Averaged LaTeX table successfully written to {averaged_output_file}")
    except IOError as e:
        logger.error(f"Failed to write averaged LaTeX table to {averaged_output_file}: {e}")

    logger.info("All LaTeX table generation complete.")


if __name__ == '__main__':
    user_scores_base_dir = os.getenv("SCORES_PATH", "data/scores")

    prompt_eng_results_file = os.path.join(
        user_scores_base_dir,
        "prompt_engineering/prompt_engineering_summaries_llama3_1b_NEWTS_train_250_articles_20250511_010741.json"
    )
    readability_steer_file = os.path.join(
        user_scores_base_dir,
        "readability_vectors/readability_summaries_llama3_1b_NEWTS_test_100_articles_words_False_20250430_180500.json"
    )
    sentiment_steer_file = os.path.join(
        user_scores_base_dir,
        "sentiment_vectors/sentiment_summaries_llama3_1b_NEWTS_train_250_articles_sentiment_sentences_20250426_003933.json"
    )
    topic_steer_file = os.path.join(
        user_scores_base_dir,
        "topic_vectors/topic_summaries_llama3_1b_NEWTS_train_250_articles_topic_words_20250429_031304.json"
    )
    toxicity_steer_file = os.path.join(
        user_scores_base_dir,
        "toxicity_vectors/toxicity_summaries_llama3_1b_NEWTS_test_100_articles_words_False_20250430_233121.json"
    )

    latex_output_dir = "data/plots/output_tables" 

    # --- Create dummy JSON files for testing if they don't exist ---
    # This helps in running the script even if actual data files are not present.
    # The dummy data aims to have some values for all configured metrics.
    dummy_files_created = False
    files_to_check = {
        "prompt_eng": prompt_eng_results_file,
        "readability_steer": readability_steer_file,
        "sentiment_steer": sentiment_steer_file,
        "topic_steer": topic_steer_file,
        "toxicity_steer": toxicity_steer_file
    }

    minimal_article_data_template = {
        # Condition keys used by prompt engineering
        "neutral": {},
        "topic_tid1_encouraged": {},
        "topic_tid2_encouraged": {}, # Discourage topic prompt condition
        "sentiment_positive_encouraged": {},
        "sentiment_negative_encouraged": {},
        "readability_simple_encouraged": {},
        "readability_complex_encouraged": {},
        "toxicity_encouraged": {},
        "toxicity_avoided": {},
        # Condition keys used by steering (must match steering_strength_keys)
        "-2": {}, "-1": {}, "1": {}, "2": {}
    }

    dummy_data_content = {"scored_summaries": {}}
    for i in range(5): # Create 5 dummy articles
        article_id = f"article_{i}"
        dummy_data_content["scored_summaries"][article_id] = json.loads(json.dumps(minimal_article_data_template)) # Deep copy
        
        for condition_key in dummy_data_content["scored_summaries"][article_id]:
            condition_data_block = {}
            # Populate with some varied data for each configured metric
            for bhv_name, bhv_config in TABLE_GENERATOR_CONFIG.items():
                for sub_m_conf in bhv_config["sub_metrics"]:
                    path_parts = sub_m_conf["path"].split('.')
                    current_level = condition_data_block
                    for k_idx, part in enumerate(path_parts):
                        if k_idx == len(path_parts) - 1: # Last part is the metric name
                            # Assign a pseudo-random value based on article, condition, and metric
                            base_val = hash(f"{article_id}_{condition_key}_{sub_m_conf['name']}") % 100 / 10.0
                            if "readability" in sub_m_conf["path"] and "deberta" in sub_m_conf["path"]:
                                current_level[part] = base_val + 5 # Shift DeBERTa a bit
                            elif "sentiment" in sub_m_conf["path"] and "vader" in sub_m_conf["path"]:
                                current_level[part] = (base_val / 5.0) - 1.0 # Vader scale -1 to 1
                            else:
                                current_level[part] = base_val
                        else: # Create nested dict
                            current_level = current_level.setdefault(part, {})
            dummy_data_content["scored_summaries"][article_id][condition_key] = condition_data_block


    for key, file_path in files_to_check.items():
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name): # Ensure dir_name is not empty
            os.makedirs(dir_name, exist_ok=True)
        if not os.path.exists(file_path):
            logger.warning(f"Dummy file being created for testing: {file_path}")
            try:
                with open(file_path, 'w') as f:
                    json.dump(dummy_data_content, f, indent=2)
                dummy_files_created = True
            except Exception as e:
                logger.error(f"Could not create dummy file {file_path}: {e}")
    if dummy_files_created:
        logger.warning("Dummy JSON files were created for testing. Please use your actual data files for correct results.")


    generate_all_latex_tables(
        prompt_eng_file_path=prompt_eng_results_file,
        topic_steering_file_path=topic_steer_file,
        sentiment_steering_file_path=sentiment_steer_file,
        readability_steering_file_path=readability_steer_file,
        toxicity_steering_file_path=toxicity_steer_file,
        output_dir=latex_output_dir
    )