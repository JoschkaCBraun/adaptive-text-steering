"""
readability_scorer.py

Evaluate text readability with two pretrained Transformer models and return their raw
regression scores.

* **DistilBERT Kaggle (CommonLit)** – outputs a continuous score centred around 0.  
  Positive ⇒ easier, negative ⇒ harder, roughly matching the CommonLit target scale
  (≈ –5 hard → +5 easy).
* **DeBERTa‑v3 Base Readability v2** – outputs an estimated U.S. grade‑level (1 ≈ 1st grade,
  18 ≈ college freshman).  Higher ⇒ more complex.

The script mirrors the structure of `sentiment_scorer.py`, but swaps in these models and
runs everything on the device returned by `src.utils.get_device()` (CUDA, MPS, or CPU).
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import sys
import logging
from typing import Dict

# ---------------------------------------------------------------------------
# Third‑party imports
# ---------------------------------------------------------------------------
from transformers import pipeline as transformer_pipeline  # alias
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from config.score_and_plot_config import ScoreAndPlotConfig
from src.utils import get_device  # <- your helper that chooses cuda/mps/cpu

# ---------------------------------------------------------------------------
# Model names (fixed here so ScoreAndPlotConfig stays untouched)
# ---------------------------------------------------------------------------
DISTILBERT_MODEL_NAME = "Tymoteusz/distilbert-base-uncased-kaggle-readability"
DEBERTA_MODEL_NAME = "agentlans/deberta-v3-base-readability-v2"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve main device once at import time so every loader uses the same one.
# `get_device()` may return a `torch.device` or a str such as "cuda", "mps", "cpu".
# ---------------------------------------------------------------------------
RAW_DEVICE = get_device()
DEVICE = (
    RAW_DEVICE if isinstance(RAW_DEVICE, torch.device) else torch.device(str(RAW_DEVICE))
)
logger.info("Selected device from get_device(): %s", DEVICE)

# ---------------------------------------------------------------------------
# Helper: build device arg for 🤗 pipeline (int for CUDA, str/torch.device for others)
# ---------------------------------------------------------------------------

def _pipeline_device_arg(dev: torch.device):
    if dev.type == "cuda":
        # pipeline expects an int index for CUDA; default to 0 if unspecified
        return dev.index if dev.index is not None else 0
    # For CPU or MPS we can pass the torch.device (supported since 🤗 v4.34).
    return dev

# ---------------------------------------------------------------------------
# Model‑loading helper functions
# ---------------------------------------------------------------------------

def load_distilbert_readability_pipeline(config: ScoreAndPlotConfig):
    """Load the DistilBERT readability model via 🤗 pipeline."""
    try:
        pipe = transformer_pipeline(
            "text-classification",
            model=DISTILBERT_MODEL_NAME,
            truncation=True,
            device=_pipeline_device_arg(DEVICE),
        )
        # Ensure internal .device is correct (pipeline may cast int→cuda idx).
        logger.info("DistilBERT readability pipeline loaded on %s.", DEVICE)
        return pipe
    except Exception as e:
        logger.error(
            "Failed to load DistilBERT readability model %s: %s",
            DISTILBERT_MODEL_NAME,
            e,
            exc_info=True,
        )
        raise


def load_deberta_readability_model(_: ScoreAndPlotConfig):
    """Load the DeBERTa‑v3 readability model (tokenizer + model to device)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_NAME)
        model.to(DEVICE)
        logger.info("DeBERTa readability model loaded on %s.", DEVICE)
        return {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        logger.error(
            "Failed to load DeBERTa readability model %s: %s",
            DEBERTA_MODEL_NAME,
            e,
            exc_info=True,
        )
        raise

# ---------------------------------------------------------------------------
# Readability scorer class
# ---------------------------------------------------------------------------

class ReadabilityScorer:
    """Load both readability models once and reuse them for multiple texts."""

    def __init__(self, config: ScoreAndPlotConfig):
        self.config = config
        logger.info("Initializing ReadabilityScorer …")

        # --- load models ---
        self.distilbert_pipeline = load_distilbert_readability_pipeline(self.config)
        deberta_assets = load_deberta_readability_model(self.config)
        self.deberta_tokenizer = deberta_assets["tokenizer"]
        self.deberta_model = deberta_assets["model"]

        logger.info("ReadabilityScorer initialized successfully.")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _analyze_distilbert(self, text: str) -> float:
        result = self.distilbert_pipeline(text)[0]
        # Pipeline with regression head → `score` holds the single float.
        return float(result["score"])

    def _analyze_deberta(self, text: str) -> float:
        inputs = self.deberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            logits = self.deberta_model(**inputs).logits.squeeze().cpu()
        return float(logits.item())

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------

    def get_readability_scores(self, text: str) -> Dict[str, float]:
        """Return raw readability scores from both models for a given text."""
        truncated = text[: self.config.max_tokens_for_sentiment_classification]
        try:
            dist_score = self._analyze_distilbert(truncated)
            deb_score = self._analyze_deberta(truncated)
            return {"distilbert": dist_score, "deberta": deb_score}
        except Exception as e:
            logger.error(
                "Failed to analyze readability for '%s…': %s",
                truncated[:50],
                e,
                exc_info=True,
            )
            return {"distilbert": 0.0, "deberta": 0.0}

# ---------------------------------------------------------------------------
# Demo / CLI execution
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting readability analysis script (Class‑Based Approach).")

    cfg = ScoreAndPlotConfig()
    logger.info(
        "Using Configuration – Device: %s | MaxTokens: %s",
        DEVICE,
        cfg.max_tokens_for_sentiment_classification,
    )

    try:
        scorer = ReadabilityScorer(cfg)
    except Exception:
        logger.error(
            "Could not initialize scorer. Ensure 'transformers', 'torch', and model weights are available.",
            exc_info=True,
        )
        sys.exit(1)

    sample_texts = [
        "I love you.",
        "This is a test.",
        "The cat sat on the mat.",
        "Students must complete their homework before watching television.",
        "The intricate ecosystem of the rainforest supports a diverse array of flora and fauna.",
        "Quantum mechanics describes the behavior of matter and energy at microscopic levels.",
        "Such the matter live tree array on test I calm."
    ]

    for idx, txt in enumerate(sample_texts, 1):
        scores = scorer.get_readability_scores(txt)
        print(
            f"\nText {idx}: '{txt}'\n DistilBERT score: {scores['distilbert']:.2f} | "
            f"DeBERTa grade‑level: {scores['deberta']:.2f}"
        )

    logger.info("Readability analysis script finished.")


if __name__ == "__main__":
    main()
