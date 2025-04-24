# Evaluation Module (`src/evaluation/`)

## Overview

This directory contains Python modules designed for evaluating various aspects of generated text. Each module focuses on a specific dimension of quality or content, allowing for a comprehensive assessment.

The goal is to provide independent, reusable "scorer" classes that can be easily integrated into evaluation pipelines or used for standalone analysis.

## Structure

Each `.py` file in this directory generally defines a primary "scorer" class responsible for a particular type of evaluation.

-   `topic_scorer.py`: Defines the `TopicScorer` class.
-   `sentiment_scorer.py`: Defines the `SentimentScorer` class.
-   `extrinsic_quality_scorer.py`: Defines the `ExtrinsicQualityScorer` class.
-   `intrinsic_quality_scorer.py`: Defines the `IntrinsicQualityScorer` class.
-   `__init__.py`: Makes this directory a Python package.

## Scorers

### 1. Topic Scorer (`topic_scorer.py`)

-   **Class:** `TopicScorer`
-   **Purpose:** Measures the topical focus of a given text against a specific topic ID derived from a pre-trained Latent Dirichlet Allocation (LDA) model.
-   **Methods/Metrics:** Calculates topic scores using different internal methods:
    -   `'dict'`: Based on LDA topic distribution for the document.
    -   `'tokenize'`: Based on the proportion of topic-specific tokens (from a transformer tokenizer) in the text.
    -   `'stem'`, `'lemmatize'`: Based on the proportion of stemmed/lemmatized topic words in the text.
-   **Inputs:** Text (string), Topic ID (int), Method (string).
-   **Requires:** Pre-trained LDA model (`lda.model`), Gensim dictionary (`dictionary.dic`), transformer tokenizer, TextProcessor utilities.

### 2. Sentiment Analyzer (`sentiment_scorer.py`)

-   **Class:** `SentimentScorer`
-   **Purpose:** Analyzes the sentiment polarity and intensity of a given text.
-   **Methods/Metrics:** Uses multiple approaches for robustness:
    -   Transformer-based sentiment classification (e.g., using models like `nlptown/bert-base-multilingual-uncased-sentiment` or similar specified in config) - provides a normalized score (e.g., based on star rating).
    -   VADER (Valence Aware Dictionary and sEntiment Reasoner) - provides a compound polarity score.
-   **Inputs:** Text (string).
-   **Requires:** Specified transformer sentiment model, VADER library, TextProcessor utilities (potentially for truncation).

### 3. Extrinsic Quality Scorer (`extrinsic_quality_scorer.py`)

-   **Class:** `ExtrinsicQualityScorer`
-   **Purpose:** Evaluates the quality of generated text by comparing it against one or more reference (ground-truth) texts. This assesses aspects like similarity, faithfulness, and overlap.
-   **Methods/Metrics (Examples):**
    -   ROUGE (R-1, R-2, R-L): Measures n-gram overlap.
    -   BERTScore: Measures semantic similarity using contextual embeddings.
    -   MAUVE: Measures divergence between distributions of generated and reference text.
    -   (Other reference-based metrics like BLEU, METEOR if applicable).
-   **Inputs:** Generated Text (string), Reference Text(s) (string or list of strings).
-   **Requires:** Libraries for the chosen metrics (e.g., `rouge-score`, `bert-score`, `evaluate[mauve]`), potentially pre-trained models for metrics like BERTScore.

### 4. Intrinsic Quality Scorer (`intrinsic_quality_scorer.py`)

-   **Class:** `IntrinsicQualityScorer`
-   **Purpose:** Evaluates the quality of generated text based on its own characteristics, without requiring a reference text. This assesses aspects like fluency, coherence, grammaticality, and repetition.
-   **Methods/Metrics (Examples):**
    -   Perplexity: Measures how well a language model predicts the text (requires a language model).
    -   Repetition Metrics: Detects repeated n-grams (e.g., distinct-1, distinct-2, long sequence repetition).
    -   Grammaticality Score: May use a dedicated grammar checking tool or a language model fine-tuned for grammatical error detection.
    -   Readability Scores: (e.g., Flesch-Kincaid, Gunning Fog).
    -   Coherence Measures: (More advanced, might involve discourse analysis or specialized models).
-   **Inputs:** Generated Text (string).
-   **Requires:** Libraries or models specific to the chosen metrics (e.g., `transformers` for perplexity, potentially grammar checking libraries, language models).

## Usage

Import the desired scorer class and initialize it (usually with a configuration object). Then, call its primary scoring method with the required inputs.

```python
# Example (conceptual)
from src.config.score_and_plot_config import ScoreAndPlotConfig
from src.evaluation.topic_scorer import TopicScorer
from src.evaluation.extrinsic_quality_scorer import ExtrinsicQualityScorer

# Load configuration
config = ScoreAndPlotConfig()

# Initialize scorers
topic_eval = TopicScorer(config)
extrinsic_eval = ExtrinsicQualityScorer(config) # Assuming config provides necessary paths/models

# Sample data
generated_text = "This is the summary generated by the model."
reference_text = "This is the reference summary."
topic_id = 10

# Get scores
topic_score_dict = topic_eval.get_topic_score(generated_text, topic_id, method='dict')
quality_scores = extrinsic_eval.get_scores(generated_text, reference_text) # Method name might vary

print(f"Topic Score: {topic_score_dict}")
print(f"Quality Scores: {quality_scores}")