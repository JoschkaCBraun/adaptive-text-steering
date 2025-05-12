# Standard library imports
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Union, Any, Optional

# Third-party imports
import torch
import numpy as np
from steering_vectors import SteeringVector

# Local imports
from config.experiment_config import ExperimentConfig
from src.utils.load_models_and_tokenizers import load_model_and_tokenizer
from src.train_vectors.get_steering_vector import get_steering_vector
from src.utils import generate_text_with_steering_vector
from src.utils.get_prompt import get_newts_summary_prompt, _build_prompt, get_topic_words # Assuming these are correctly importable
from src.utils.load_datasets import load_newts_dataframe, NEWTSDataset
from src.utils import load_lda # For topic modeling

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants from get_newts_summary_prompt ---
BASE_SUMMARY_INSTRUCTION = "Write a three sentence summary of the article"

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    '''Custom JSON encoder to handle numpy types.'''
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyEncoder, self).default(o)

def generate_steered_and_prompted_summaries(
    config_obj: ExperimentConfig, # Pass the config object for its internal values
    model_alias_main: str,
    load_test_set_main: bool,
    num_articles_main: int,
    representation_type_main: str,
    steering_layers_main: List[int],
    language_main: str = "en",
    pairing_type_topic_main: Optional[str] = 'against_random_topic_representation',
    pairing_type_other_main: Optional[str] = None,
    num_topic_words_main: int = 25, # Default if not in config_obj
    behavior_samples_main: int = 100 # Default if not in config_obj
) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Generates NEWTS summaries applying combined steering and prompting.
    - Topic: Encourages tid1 (+1), Neutral tid1 (0), Encourages tid2 (+1 in 'discouraging tid1' slot).
    - Other behaviors (Sentiment, Toxicity, Readability): Discouraging (-1), Neutral (0), Encouraging (+1).
    """
    tokenizer, model, device = load_model_and_tokenizer(model_alias=model_alias_main)
    max_new_tokens = config_obj.MAX_NEW_TOKENS # Get from config object
    
    # Use values from config_obj if available, otherwise use main parameters as defaults
    num_topic_words_for_prompt = getattr(config_obj, 'NUM_TOPIC_WORDS', num_topic_words_main)
    num_samples_for_vector = getattr(config_obj, 'BEHAVIOR_WORDS_NUM_SAMPLES', behavior_samples_main)
    
    model.eval()

    lda_model = load_lda()
    if lda_model is None and num_topic_words_for_prompt > 0:
        logger.warning("LDA model not loaded. Topic-focused prompts might be affected.")

    df = load_newts_dataframe(num_articles=num_articles_main, load_test_set=load_test_set_main)
    dataset = NEWTSDataset(dataframe=df)

    # Define the standard behavior types from config or hardcode
    valid_behavior_types = getattr(config_obj, 'VALID_BEHAVIOR_TYPES', ['topic', 'sentiment', 'toxicity', 'readability'])
    if 'topic' not in valid_behavior_types: # Ensure topic is always considered if this script's logic depends on it
        logger.warning("'topic' not in VALID_BEHAVIOR_TYPES from config, but this script has special logic for it.")

    # Load steering vectors for non-topic behaviors once at the beginning
    steering_vectors = {}
    for behavior_name in valid_behavior_types:
        if behavior_name != "topic":
            logger.info(f"Loading steering vector for {behavior_name}")
            try:
                steering_vectors[behavior_name] = get_steering_vector(
                    behavior_type=behavior_name,
                    model_alias=model_alias_main,
                    num_samples=num_samples_for_vector,
                    representation_type=representation_type_main,
                    steering_layers=steering_layers_main,
                    language=language_main,
                    pairing_type=pairing_type_other_main
                )
                if steering_vectors[behavior_name] is None:
                    logger.warning(f"Could not generate steering vector for {behavior_name}")
            except Exception as e:
                logger.error(f"Error getting steering vector for {behavior_name}: {e}")
                steering_vectors[behavior_name] = None

    experiment_information = {
        "experiment_name":     "steering_and_prompting_custom_topic",
        "model_alias":         model_alias_main,
        "behaviors_processed": valid_behavior_types,
        "load_test_set":       load_test_set_main,
        "num_articles":        num_articles_main,
        "max_new_tokens":      max_new_tokens,
        "representation_type": representation_type_main,
        "language":            language_main,
        "steering_layers":     steering_layers_main,
        "pairing_type_topic":  pairing_type_topic_main,
        "pairing_type_other_behaviors": pairing_type_other_main,
        "num_samples_for_vector": num_samples_for_vector,
        "num_topic_words_for_prompt": num_topic_words_for_prompt,
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results: Dict[str, Any] = {"experiment_information": experiment_information}
    generated_summaries_overall: Dict[str, Dict[str, Any]] = {}

    try:
        with torch.no_grad():
            for article_loop_idx in range(min(len(dataset), num_articles_main)):
                article_data = dataset[article_loop_idx]
                article_actual_idx = int(article_data['article_idx'])
                docId = article_data['docId']
                article_text = article_data['article']
                tid1_orig = article_data['tid1']
                tid2_orig = article_data['tid2']
                original_summary1 = article_data['summary1']
                original_summary2 = article_data['summary2']

                logger.info(f"Processing article {article_loop_idx + 1}/{num_articles_main} (ID: {article_actual_idx}, docId: {docId})")

                article_results_container: Dict[str, Any] = {
                    'docId': docId, 'article_idx': article_actual_idx, 'article_text': article_text,
                    'original_tid1': tid1_orig, 'original_tid2': tid2_orig,
                    'original_summary1': original_summary1, 'original_summary2': original_summary2,
                    'generated_behavior_summaries': {}
                }
                
                for behavior_name in valid_behavior_types:
                    logger.info(f"  Behavior: {behavior_name}")
                    
                    if behavior_name == "topic":
                        output_key_topic = "topic_custom_conditions"
                        article_results_container['generated_behavior_summaries'][output_key_topic] = {}
                        
                        current_tid1 = None
                        if tid1_orig is not None and tid1_orig != "" and not (isinstance(tid1_orig, float) and np.isnan(tid1_orig)):
                            current_tid1 = int(tid1_orig)
                        current_tid2 = None
                        if tid2_orig is not None and tid2_orig != "" and not (isinstance(tid2_orig, float) and np.isnan(tid2_orig)):
                            current_tid2 = int(tid2_orig)

                        if current_tid1 is None:
                            logger.warning(f"    tid1 ('{tid1_orig}') is invalid for article {article_actual_idx}. Skipping tid1-related topic conditions.")
                        if current_tid2 is None:
                             logger.warning(f"    tid2 ('{tid2_orig}') is invalid for article {article_actual_idx}. Skipping tid2-related topic condition (for 'discouraging' slot).")


                        # Condition 1: Encourage tid1
                        condition_name_enc_tid1 = "encouraging_tid1" # More descriptive key
                        prompt_str = f"Error: Prompt for encourage_tid1 not generated"
                        summary_str = "Error: Summary for encourage_tid1 not generated"
                        if current_tid1 is not None:
                            logger.info(f"    Sub-condition: Encourage tid1 (TID: {current_tid1})")
                            vec_tid1: Optional[SteeringVector] = None
                            try:
                                vec_tid1 = get_steering_vector(behavior_type="topic", model_alias=model_alias_main, num_samples=num_samples_for_vector,
                                                               representation_type=representation_type_main, steering_layers=steering_layers_main,
                                                               language=language_main, pairing_type=pairing_type_topic_main, tid=current_tid1)
                                if vec_tid1 is None: logger.warning(f"      Could not get steering vector for TID {current_tid1}.")
                            except Exception as e: logger.error(f"      Error getting vector for TID {current_tid1}: {e}")
                            
                            try:
                                prompt_str = get_newts_summary_prompt(article=article_text, behavior_type="topic", use_behavior_encouraging_prompt=True,
                                                                    lda=lda_model, tid=current_tid1, num_topic_words=num_topic_words_for_prompt)
                                if vec_tid1:
                                    summary_str = generate_text_with_steering_vector(model=model, tokenizer=tokenizer, prompt=prompt_str, steering_vector=vec_tid1,
                                                                                   steering_strength=1.0, device=device, max_new_tokens=max_new_tokens)
                                else: summary_str = "Error: Steering vector for tid1 missing."
                            except Exception as e: logger.error(f"      Error generating summary for encourage_tid1: {e}"); summary_str = f"Error: {str(e)}"
                        else: summary_str = "Error: tid1 invalid for this condition."
                        article_results_container['generated_behavior_summaries'][output_key_topic][condition_name_enc_tid1] = {
                            "prompt": prompt_str, "summary": summary_str, "steering_strength": 1.0, "target_tid": current_tid1,
                            "steering_vector_details": f"vector_for_tid_{current_tid1}" if current_tid1 and vec_tid1 else "None"
                        }

                        # Condition 2: Neutral for tid1
                        condition_name_neutral_tid1 = "neutral_tid1"
                        prompt_str = f"Error: Prompt for neutral_tid1 not generated"
                        summary_str = "Error: Summary for neutral_tid1 not generated"
                        if current_tid1 is not None: # Neutral still uses tid1 vector with strength 0
                            logger.info(f"    Sub-condition: Neutral tid1 (TID: {current_tid1})")
                            # vec_tid1 should be available from above if current_tid1 is not None
                            try:
                                prompt_str = get_newts_summary_prompt(article=article_text, use_behavior_encouraging_prompt=False)
                                if vec_tid1 : # Reuse vec_tid1 if available
                                    summary_str = generate_text_with_steering_vector(model=model, tokenizer=tokenizer, prompt=prompt_str, steering_vector=vec_tid1,
                                                                                   steering_strength=0.0, device=device, max_new_tokens=max_new_tokens)
                                else: summary_str = "Error: Steering vector for tid1 missing for neutral condition."
                            except Exception as e: logger.error(f"      Error generating summary for neutral_tid1: {e}"); summary_str = f"Error: {str(e)}"
                        else: summary_str = "Error: tid1 invalid for this condition."
                        article_results_container['generated_behavior_summaries'][output_key_topic][condition_name_neutral_tid1] = {
                            "prompt": prompt_str, "summary": summary_str, "steering_strength": 0.0, "target_tid": current_tid1,
                            "steering_vector_details": f"vector_for_tid_{current_tid1}" if current_tid1 and vec_tid1 else "None"
                        }

                        # Condition 3: Encourage tid2 (in "discouraging tid1" slot)
                        condition_name_enc_tid2 = "discouraging_tid1_slot_encourages_tid2"
                        prompt_str = f"Error: Prompt for encourage_tid2 not generated"
                        summary_str = "Error: Summary for encourage_tid2 not generated"
                        if current_tid2 is not None:
                            logger.info(f"    Sub-condition: Encourage tid2 (TID: {current_tid2}) for 'discouraging tid1' slot")
                            vec_tid2: Optional[SteeringVector] = None
                            try:
                                vec_tid2 = get_steering_vector(behavior_type="topic", model_alias=model_alias_main, num_samples=num_samples_for_vector,
                                                               representation_type=representation_type_main, steering_layers=steering_layers_main,
                                                               language=language_main, pairing_type=pairing_type_topic_main, tid=current_tid2)
                                if vec_tid2 is None: logger.warning(f"      Could not get steering vector for TID {current_tid2}.")
                            except Exception as e: logger.error(f"      Error getting vector for TID {current_tid2}: {e}")
                            
                            try:
                                prompt_str = get_newts_summary_prompt(article=article_text, behavior_type="topic", use_behavior_encouraging_prompt=True,
                                                                    lda=lda_model, tid=current_tid2, num_topic_words=num_topic_words_for_prompt)
                                if vec_tid2:
                                    summary_str = generate_text_with_steering_vector(model=model, tokenizer=tokenizer, prompt=prompt_str, steering_vector=vec_tid2,
                                                                                   steering_strength=1.0, device=device, max_new_tokens=max_new_tokens)
                                else: summary_str = "Error: Steering vector for tid2 missing."
                            except Exception as e: logger.error(f"      Error generating summary for encourage_tid2: {e}"); summary_str = f"Error: {str(e)}"
                        else: summary_str = "Error: tid2 invalid for this condition."
                        article_results_container['generated_behavior_summaries'][output_key_topic][condition_name_enc_tid2] = {
                            "prompt": prompt_str, "summary": summary_str, "steering_strength": 1.0, "target_tid": current_tid2, "note": "This slot encourages tid2",
                            "steering_vector_details": f"vector_for_tid_{current_tid2}" if current_tid2 and vec_tid2 else "None"
                        }
                    
                    else: # For sentiment, toxicity, readability - Standard 3 conditions
                        article_results_container['generated_behavior_summaries'][behavior_name] = {}
                        steering_vec_other = steering_vectors.get(behavior_name)
                        
                        conditions_map_other = {"discouraging": -1.0, "neutral": 0.0, "encouraging": 1.0}
                        for condition, strength in conditions_map_other.items():
                            prompt_str = f"Error: Prompt not generated for {condition} {behavior_name}"
                            summary_str = f"Error: Summary not generated ({'vector missing' if steering_vec_other is None else 'generation error'})"
                            use_enc_prompt_flag = (condition != "neutral")
                            
                            try:
                                if behavior_name == "sentiment":
                                    enc_positive = (condition == "encouraging")
                                    prompt_str = get_newts_summary_prompt(article=article_text, behavior_type="sentiment", use_behavior_encouraging_prompt=use_enc_prompt_flag, encourage_positive_sentiment=enc_positive)
                                elif behavior_name == "toxicity":
                                    enc_toxic = (condition == "encouraging")
                                    prompt_str = get_newts_summary_prompt(article=article_text, behavior_type="toxicity", use_behavior_encouraging_prompt=use_enc_prompt_flag, encourage_toxicity=enc_toxic)
                                elif behavior_name == "readability":
                                    enc_simple = (condition == "encouraging")
                                    prompt_str = get_newts_summary_prompt(article=article_text, behavior_type="readability", use_behavior_encouraging_prompt=use_enc_prompt_flag, encourage_simplicity=enc_simple)
                                
                                if steering_vec_other:
                                    summary_str = generate_text_with_steering_vector(model=model, tokenizer=tokenizer, prompt=prompt_str, steering_vector=steering_vec_other,
                                                                                   steering_strength=strength, device=device, max_new_tokens=max_new_tokens)
                            except Exception as e:
                                logger.error(f"      Error generating summary for {condition} {behavior_name}: {e}"); summary_str = f"Error: {str(e)}"
                            
                            article_results_container['generated_behavior_summaries'][behavior_name][condition] = {
                                "prompt": prompt_str, "summary": summary_str, "steering_strength": strength,
                                "steering_vector_details": f"vector_for_{behavior_name}" if steering_vec_other else "None"
                            }
                
                generated_summaries_overall[str(article_actual_idx)] = article_results_container
                logger.info(f"  Completed all behaviors for article {article_loop_idx + 1}/{num_articles_main} (ID: {article_actual_idx})")

        results['generated_summaries'] = generated_summaries_overall
        logger.info(f"Generated summaries for {len(generated_summaries_overall)} articles.")

        dataset_name_suffix = "test" if load_test_set_main else "train_val"
        _save_results(
            results=results, model_alias=model_alias_main, dataset_name_suffix=dataset_name_suffix,
            num_articles=len(generated_summaries_overall), representation_type=representation_type_main,
            experiment_folder="steering_and_prompting",
            base_filename_prefix="custom_topic_all_behaviors"
        )
        return results

    except Exception as e:
        logger.error(f"Fatal error in generate_steered_and_prompted_summaries: {e}", exc_info=True)
        if 'generated_summaries' not in results: results['generated_summaries'] = generated_summaries_overall
        if generated_summaries_overall: # Save partial if any generated
            dataset_name_suffix = "test" if load_test_set_main else "train_val"
            _save_results(
                results=results, model_alias=model_alias_main, dataset_name_suffix=dataset_name_suffix,
                num_articles=len(generated_summaries_overall), representation_type=representation_type_main,
                experiment_folder="steering_and_prompting_PARTIAL",
                base_filename_prefix="custom_topic_all_behaviors_PARTIAL"
            )
        raise

def _save_results(
    results: Dict[str, Any], model_alias: str, dataset_name_suffix: str, num_articles: int,
    representation_type: str, experiment_folder: str, base_filename_prefix: str
) -> None:
    NEWTS_SUMMARIES_PATH = os.getenv("NEWTS_SUMMARIES_PATH", "./outputs/newts_summaries")
    output_dir = os.path.join(NEWTS_SUMMARIES_PATH, experiment_folder)
    os.makedirs(output_dir, exist_ok=True)

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{base_filename_prefix}_summaries_{model_alias}_NEWTS_{dataset_name_suffix}_"
            f"{num_articles}articles_{representation_type}_{timestamp}.json"
        )
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)

def main() -> None:
    # Hardcoded configuration values for main execution
    MODEL_ALIAS = "llama3_1b"  # Or your desired model
    LOAD_TEST_SET = False
    NUM_ARTICLES_TO_PROCESS = 2 # Small number for testing, increase as needed
    DEFAULT_REPRESENTATION_TYPE = "words"
    DEFAULT_STEERING_LAYERS = [8]
    DEFAULT_LANGUAGE = "en"
    PAIRING_TYPE_TOPIC = 'against_random_topic_representation'
    PAIRING_TYPE_OTHER = None # For non-topic behaviors

    # Values that might come from ExperimentConfig or be defaults
    NUM_TOPIC_WORDS = 25
    BEHAVIOR_WORDS_NUM_SAMPLES = 100

    # Instantiate ExperimentConfig - it might still be used for some internal constants by helper functions
    # or for MAX_NEW_TOKENS, VALID_BEHAVIOR_TYPES etc.
    try:
        config = ExperimentConfig()
        # Override or set defaults if ExperimentConfig doesn't have them or for consistency
        config.MAX_NEW_TOKENS = getattr(config, 'MAX_NEW_TOKENS', 150) # Default if not in config
        config.NUM_TOPIC_WORDS = getattr(config, 'NUM_TOPIC_WORDS', NUM_TOPIC_WORDS)
        config.BEHAVIOR_WORDS_NUM_SAMPLES = getattr(config, 'BEHAVIOR_WORDS_NUM_SAMPLES', BEHAVIOR_WORDS_NUM_SAMPLES)
        config.VALID_BEHAVIOR_TYPES = getattr(config, 'VALID_BEHAVIOR_TYPES', ['topic', 'sentiment', 'toxicity', 'readability'])

    except Exception as e:
        logger.error(f"Error initializing ExperimentConfig: {e}. Using hardcoded fallbacks for some values.")
        # Create a dummy config or a simple namespace if ExperimentConfig is problematic
        class DummyConfig:
            MAX_NEW_TOKENS = 150
            NUM_TOPIC_WORDS = NUM_TOPIC_WORDS
            BEHAVIOR_WORDS_NUM_SAMPLES = BEHAVIOR_WORDS_NUM_SAMPLES
            VALID_BEHAVIOR_TYPES = ['topic', 'sentiment', 'toxicity', 'readability']
        config = DummyConfig()


    logger.info("Starting combined steering and prompting experiment with custom topic handling.")
    logger.info(f"Configuration: Model={MODEL_ALIAS}, Dataset={'Test' if LOAD_TEST_SET else 'Train/Val'}, NumArticles={NUM_ARTICLES_TO_PROCESS}")
    logger.info(f"RepresentationType={DEFAULT_REPRESENTATION_TYPE}, SteeringLayers={DEFAULT_STEERING_LAYERS}, Language={DEFAULT_LANGUAGE}")
    logger.info(f"NumTopicWordsForPrompt={config.NUM_TOPIC_WORDS}, BehaviorSamples={config.BEHAVIOR_WORDS_NUM_SAMPLES}")
    logger.info(f"PairingType (Topic)='{PAIRING_TYPE_TOPIC}', PairingType (Other Behaviors)='{PAIRING_TYPE_OTHER}'")


    if not os.getenv("NEWTS_SUMMARIES_PATH"):
        logger.warning("NEWTS_SUMMARIES_PATH environment variable not set. Results will default to ./outputs/newts_summaries.")
    
    if not all(b_type in ["topic", "sentiment", "toxicity", "readability"] for b_type in config.VALID_BEHAVIOR_TYPES):
        logger.warning(f"config.VALID_BEHAVIOR_TYPES ({config.VALID_BEHAVIOR_TYPES}) contains unrecognized types. Expected subsets of ['topic', 'sentiment', 'toxicity', 'readability']. Processing only recognized types.")


    try:
        generate_steered_and_prompted_summaries(
            config_obj=config, # Pass the config instance
            model_alias_main=MODEL_ALIAS,
            load_test_set_main=LOAD_TEST_SET,
            num_articles_main=NUM_ARTICLES_TO_PROCESS,
            representation_type_main=DEFAULT_REPRESENTATION_TYPE,
            steering_layers_main=DEFAULT_STEERING_LAYERS,
            language_main=DEFAULT_LANGUAGE,
            pairing_type_topic_main=PAIRING_TYPE_TOPIC,
            pairing_type_other_main=PAIRING_TYPE_OTHER,
            num_topic_words_main=config.NUM_TOPIC_WORDS, # Pass explicitly
            behavior_samples_main=config.BEHAVIOR_WORDS_NUM_SAMPLES # Pass explicitly
        )
        logger.info("Experiment finished successfully.")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()