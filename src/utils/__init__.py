# utils/__init__.py
from .compute_dimensionality_reduction import (
    compute_binary_pca_projection,
    compute_binary_lda_projection,
)

from .compute_steering_vectors import (
    compute_contrastive_activations,
    compute_contrastive_activations_dict,
    compute_contrastive_steering_vector,
    compute_contrastive_steering_vector_dict,
    compute_non_contrastive_steering_vector,
    compute_in_context_vector,
)

from .compute_properties_of_activations import (
    compute_pairwise_cosine_similarity,
    compute_pairwise_cosine_similarity_dict,
    compute_many_against_one_cosine_similarity,
    compute_many_against_one_cosine_similarity_dict,
    compute_l2_norms,
    compute_l2_norms_dict,
)

from .get_basic import (
    get_device,
    get_path,
)

from .load_stored_anthropic_evals_activations_and_logits import (
    load_anthropic_evals_activations_and_logits,
    load_paired_activations_and_logits
)

from .plotting_functions.plot_results_as_violine_plots import (
    plot_all_results,
)

from .logits_to_probs_utils import (
    compute_probability_differences,
    get_token_probability,
    map_answer_tokens_to_ids,
    extract_token_type,
    get_answer_token_ids,
)

from .generate_prompts_for_anthropic_evals_dataset import (
    generate_prompt_and_anwers,
)

from .load_datasets import (
    load_newts_dataset,
)

from .load_models_and_tokenizers import (
    load_model_and_tokenizer,
    load_tokenizer,
    get_model_info,
)

from .lda_utils import (
    load_lda_and_dictionary,
    load_dictionary,
    load_lda,
    get_topic_words,
)

from .topic_training_samples_utils import (
    save_topic_training_samples,
    load_topic_training_samples,
)

from ._validate_custom_datatypes import (
    Layer,
    Activation,
    SteeringVector,
    ActivationList,
    ActivationListDict,
    SteeringVectorDict,
)

__all__ = [
    # Types
    "Layer",
    "Activation",
    "SteeringVector",
    "ActivationList",
    "ActivationListDict",
    "SteeringVectorDict",
    # compute dimensionality reduction
    "compute_binary_pca_projection",
    "compute_binary_lda_projection",
    # compute steering vectors
    "compute_contrastive_activations",
    "compute_contrastive_activations_dict",
    "compute_contrastive_steering_vector",
    "compute_contrastive_steering_vector_dict",
    "compute_non_contrastive_steering_vector",
    "compute_in_context_vector",
    # compute properties of activations
    "compute_pairwise_cosine_similarity",
    "compute_pairwise_cosine_similarity_dict",
    "compute_many_against_one_cosine_similarity",
    "compute_many_against_one_cosine_similarity_dict",
    "compute_l2_norms",
    "compute_l2_norms_dict",
    # load stored anthropic evals activations and logits
    "load_anthropic_evals_activations_and_logits",
    "load_paired_activations_and_logits",
    # generate prompts for anthropic evals dataset
    "generate_prompt_and_anwers",
    # plotting functions
    "plot_all_results",
    # logits to probs utils
    "compute_probability_differences",
    "get_token_probability",
    "map_answer_tokens_to_ids",
    "extract_token_type",
    "get_answer_token_ids",
    # get basic
    "get_device",
    "get_path",
    # load datasets
    "load_newts_dataset",
    # load models and tokenizers
    "load_model_and_tokenizer",
    "load_tokenizer",
    "get_model_info",
    # lda utils
    "load_lda_and_dictionary",
    "load_dictionary",
    "load_lda",
    "get_topic_words",
    # topic training samples
    "save_topic_training_samples",
    "load_topic_training_samples",
]