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
)

from .compute_properties_of_activations import (
    compute_pairwise_cosine_similarity,
    compute_pairwise_cosine_similarity_dict,
    compute_many_against_one_cosine_similarity,
    compute_many_against_one_cosine_similarity_dict,
    compute_l2_norms,
    compute_l2_norms_dict,
)

from .generation_utils import (
    generate_text_with_steering_vector,
)

from .get_basic import (
    get_device,
    get_path,
)

from .get_prompt import (
    get_newts_summary_topic_prompt,
    get_newts_summary_sentiment_prompt,
)

from .plotting_functions.plot_results_as_violine_plots import (
    plot_all_results,
)

from .load_datasets import (
    load_newts_dataloader,
    load_newts_dataframe,
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

from .sentiment_utils import (
    load_sentiment_vector_training_samples,
    get_sentiment_representation_file_name,
)

from .topic_training_samples_utils import (
    save_topic_representations,
    load_topic_representations,
    save_topic_vector_training_samples,
    load_topic_vector_training_samples,
    get_topic_vector_file_path,
    get_topic_vector_file_name,
    get_topic_vectors_folder_name,
    save_topic_vector,
    load_topic_vector,
)

from .validate_inputs import (
    validate_topic_representation_type,
    validate_pairing_type,
    validate_language,
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
    # compute properties of activations
    "compute_pairwise_cosine_similarity",
    "compute_pairwise_cosine_similarity_dict",
    "compute_many_against_one_cosine_similarity",
    "compute_many_against_one_cosine_similarity_dict",
    "compute_l2_norms",
    "compute_l2_norms_dict",
    # generation utils
    "generate_text_with_steering_vector",
    # get basic
    "get_device",
    "get_path",
    # get prompt
    "get_newts_summary_topic_prompt",
    "get_newts_summary_sentiment_prompt",
    # load datasets
    "load_newts_dataloader",
    "load_newts_dataframe",
    # load models and tokenizers
    "load_model_and_tokenizer",
    "load_tokenizer",
    "get_model_info",
    # lda utils
    "load_lda_and_dictionary",
    "load_dictionary",
    "load_lda",
    # "get_topic_words",
    # sentiment utils
    "load_sentiment_vector_training_samples",
    "get_sentiment_representation_file_name",
    # topic representations
    "save_topic_representations",
    "load_topic_representations",
    "save_topic_vector_training_samples",
    "load_topic_vector_training_samples",
    "get_topic_vector_file_path",
    "get_topic_vector_file_name",
    "get_topic_vectors_folder_name",
    "save_topic_vector",
    "load_topic_vector",
    # validate inputs
    "validate_topic_representation_type",
    "validate_pairing_type",
    "validate_language",
]