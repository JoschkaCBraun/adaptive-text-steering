'''
score_list_of_texts.py
Given a list of texts, this script iterates through each text under different
conditions (varying topic IDs and reference texts) and scores it using
intrinsic, extrinsic, sentiment, and topic metrics via the unified Scorer class.

The results for each scoring scenario are stored in a list of dictionaries
and saved to a JSON file.
'''

# Standard library imports
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List

# Local application imports
from src.evaluation.scorer import Scorer

# --- Setup Logging ---
# Keep existing logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Constants ---
# Keep existing constants
INPUT_FILE = Path("data/results/list_of_texts.json")
OUTPUT_FILE = Path("data/results/scores/scores_list_of_texts.json")

REFERENCE_TEXTS_TO_USE: List[Optional[str]] = [
    None, # Scenario without extrinsic scoring
    "Premier league is the most paying football league in the world with each player earning an average of 2.27m euros. In terms of best paying teams Liverpool is at the 20th position while Chelsea is at the 10th position. Manchester United ranks in the 8th position with each player earning around 4.3 million euro per year.",
    "Manchester city players earn the largest amounts of money in the whole Premier football league. Arsenal have advanced to the finals in the FA cup after beating Wigan. City football players were earning around 1.5 m euro before Sheik Mansour arrived in 2008."
]

TIDS_TO_USE: List[Optional[int]] = [
    None, # Scenario without topic scoring
    175,
    110,
    12,
    50
]

# --- Core Scoring Function ---
# This function is now removed as its logic is inside the Scorer class method.

# --- Main Execution Logic ---

def run_scoring():
    """Loads data, initializes the unified scorer, runs nested loops, and saves results."""
    logger.info("--- Starting Text Scoring Script ---")

    # --- 1. Load Input Data ---
    logger.info(f"Loading generated texts from: {INPUT_FILE}")
    if not INPUT_FILE.is_file():
        logger.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            generated_texts: List[str] = json.load(f)
        if not isinstance(generated_texts, list) or not all(isinstance(item, str) for item in generated_texts):
             logger.error(f"Input file {INPUT_FILE} does not contain a list of strings.")
             sys.exit(1)
        logger.info(f"Loaded {len(generated_texts)} generated texts.")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {INPUT_FILE}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading input file {INPUT_FILE}: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Initialize Unified Scorer ---
    # The Scorer class handles its own configuration internally.
    logger.info("Initializing the unified Scorer...")
    try:
        scorer = Scorer() # Initialize the main scorer
        logger.info("Unified Scorer initialized successfully.")
    except Exception as e:
        # Catch errors during scorer initialization (e.g., model loading failed)
        logger.error(f"Fatal error during Scorer initialization: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Run Scoring Loops ---
    all_results = []
    total_combinations = len(TIDS_TO_USE) * len(REFERENCE_TEXTS_TO_USE) * len(generated_texts)
    logger.info(f"Starting scoring for {total_combinations} total combinations (TIDs x Refs x Texts)...")

    # Use nested loops as requested
    # Outer loop: Topic IDs (including None)
    for tid in TIDS_TO_USE:
        # Middle loop: Reference Texts (including None)
        for ref_text in REFERENCE_TEXTS_TO_USE:
            logger.debug(f"--- Processing combination: TID={tid}, Reference={ref_text is not None} ---")
            # Inner loop: Generated Texts
            # Use tqdm for progress bar on the innermost loop
            for gen_text in generated_texts:
                try:
                    # Call the unified scorer's method
                    # It handles conditional scoring internally based on tid and ref_text presence.
                    # It uses default topic_method='lemmatize' and distinct_n=2 unless specified otherwise.
                    single_result = scorer.score_individual_text(
                        text=gen_text,
                        tid1=tid,
                        reference_text1=ref_text
                        # Can optionally pass topic_method='...' or distinct_n=... here if needed
                    )

                    all_results.append(single_result)

                except Exception as e:
                    # Catch potential errors from the scorer.score_individual_text call
                    # The Scorer method has internal logging, but we catch here too for robustness.
                    logger.error(f"Scoring failed for text '{gen_text[:50]}...' with TID={tid}, Ref={ref_text is not None}. Error: {e}", exc_info=False) # exc_info=False to keep logs concise
                    # Optionally append an error record:
                    all_results.append({
                        "text": gen_text, # Changed from "generated_text" to match Scorer output
                        "error": str(e),
                        "tid1_used": tid,
                        "reference_text1_used": ref_text # Store ref text even on error
                    })
                    # Continue to the next text/combination

    # --- 4. Save Results ---
    logger.info(f"Scoring complete. Collected {len(all_results)} results.")
    logger.info(f"Saving results to: {OUTPUT_FILE}")

    try:
        # Ensure the output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Save the results list as JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False) # Use indent for readability
        logger.info("Results saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save results to {OUTPUT_FILE}: {e}", exc_info=True)

    logger.info("--- Text Scoring Script Finished ---")


if __name__ == "__main__":
    # Assuming scorer.py is in src/evaluation/
    # Adjust sys.path if necessary, or ensure your project structure allows the import
    # Example: Add project root to path if running script directly
    # project_root = Path(__file__).resolve().parent.parent # Adjust based on your structure
    # sys.path.insert(0, str(project_root))
    # from src.evaluation.scorer import Scorer # Now the import should work

    run_scoring()