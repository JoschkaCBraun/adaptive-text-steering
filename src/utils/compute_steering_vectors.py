'''
This module contains functions to compute steering vectors from neural network activations.

The module provides three types of steering vector computation:
1. Contrastive: Computed as the mean difference between positive and negative activations
2. Non-contrastive: Computed from positive activations only
3. In-Context Vectors (ICV): Computed using PCA on the difference between positive and negative activations
'''
from typing import List, Dict
import torch
import src.utils._validate_custom_datatypes as vcd
from src.utils._validate_custom_datatypes import (
    SteeringVector,
    ActivationList,
    ActivationListDict,
    SteeringVectorDict,
)
import src.utils.compute_dimensionality_reduction as cdr

def compute_contrastive_activations(
    positive_activations: ActivationList,
    negative_activations: ActivationList,
    device: torch.device = torch.device("cpu")
) -> ActivationList:
    """
    Computes the contrastive activations by subtracting each negative activation from its corresponding positive activation.

    Args:
        positive_activations (ActivationList): A list of tensors representing positive activations.
        negative_activations (ActivationList): A list of tensors representing negative activations.
        device (torch.device): The device on which to perform the computations.

    Returns:
        ActivationList: A list of tensors where each tensor is the difference (positive - negative).
    """
    vcd._validate_activation_list_compatibility(
        activations1=positive_activations,
        activations2=negative_activations
    )
    
    contrastive_activations = []
    for pos, neg in zip(positive_activations, negative_activations):
        pos_act = pos.to(device)
        neg_act = neg.to(device)
        contrastive_activations.append(pos_act - neg_act)
    
    return contrastive_activations

def compute_contrastive_activations_dict(
        positive_activations: ActivationListDict,
        negative_activations: ActivationListDict,
        device: torch.device = torch.device("cpu")) -> ActivationListDict:
    """
    Computes the difference between positive and negative activations for each layer.
    This function calculates pos - neg activations without averaging, returning the raw
    contrastive activations for each sample.

    Args:
        positive_activations (ActivationListDict): A dictionary containing positive activations for
            each layer. Each value is a list of tensors of shape (hidden_size).
        negative_activations (ActivationListDict): A dictionary containing negative activations for
            each layer. Each value is a list of tensors of shape (hidden_size).
        device (torch.device): The device to perform computations on.

    Returns:
        ActivationListDict: A dictionary containing the contrastive activations for each layer.
            Keys are layer indices (int) and values are lists of tensors of shape (hidden_size).
            Each tensor represents the difference between corresponding positive and negative
            activations.
    """
    vcd._validate_activation_list_dict_to_activation_list_dict_compatibility(
        activations1=positive_activations, activations2=negative_activations)
    
    contrastive_activations = {}
    for layer in positive_activations:
        layer_contrastive_acts = []
        for pos, neg in zip(positive_activations[layer], negative_activations[layer]):
            pos_act = pos.to(device)
            neg_act = neg.to(device)
            layer_contrastive_acts.append(pos_act - neg_act)
        contrastive_activations[layer] = layer_contrastive_acts
    
    vcd._validate_activation_list_dict(
        layer_dict=contrastive_activations,
        expected_shape=next(iter(positive_activations.values()))[0].shape)
    
    return contrastive_activations

def compute_contrastive_steering_vector(
        positive_activations: ActivationList,
        negative_activations: ActivationList,
        device: torch.device = torch.device("cpu")) -> SteeringVector:
    """
    This function calculates the difference between two lists of positive and
    negative activations and returns the mean difference as the steering vector.

    Args:
        positive_activations (ActivationList): A list of tensors of shape (hidden_size)
            containing positive activations.
        negative_activations (ActivationList): A list of tensors of shape (hidden_size)
            containing negative activations.
        device (torch.device): The device to perform computations on.

    Returns:
        SteeringVector: A tensor of shape (hidden_size) representing the steering vector
            computed as the mean difference between positive and negative activations.
    """
    contrastive_activations = compute_contrastive_activations(
        positive_activations=positive_activations, negative_activations=negative_activations,
        device=device)
    
    steering_vector = torch.mean(torch.stack(contrastive_activations), dim=0)
    
    return steering_vector
        
def compute_contrastive_steering_vector_dict(
        positive_activations: ActivationListDict, negative_activations: ActivationListDict,
        device: torch.device = torch.device("cpu")) -> SteeringVectorDict:
    """
    This function calculates the difference between positive and negative activations
    for each layer and returns the mean difference as the steering vector.

    Args:
        positive_activations (ActivationListDict): A dictionary containing positive activations for
            each layer. Each value is a list of tensors of shape (hidden_size).
        negative_activations (ActivationListDict): A dictionary containing negative activations for
            each layer. Each value is a list of tensors of shape (hidden_size).
        device (torch.device): The device to perform computations on.

    Returns:
        SteeringVectors: A dictionary containing the computed steering vectors for each layer.
            Keys are layer indices (int) and values are tensors of shape (hidden_size,).
    """
    contrastive_activations_dict_list = compute_contrastive_activations_dict(
        positive_activations, negative_activations, device)

    steering_vectors = {}
    for layer, contrastive_activations_list in contrastive_activations_dict_list.items():
        steering_vectors[layer] = torch.mean(torch.stack(contrastive_activations_list), dim=0)

    vcd._validate_steering_vector_dict(
        layer_dict=steering_vectors, 
        expected_shape=next(iter(positive_activations.values()))[0].shape)
    return steering_vectors

def compute_non_contrastive_steering_vector(
        positive_activations: ActivationListDict, device: torch.device = torch.device("cpu")
        ) -> SteeringVectorDict:
    """
    Compute the non-contrastive steering vector from positive activations.

    Args:
        positive_activations (ActivationListDict): A dictionary containing positive activations for
            each layer. Each value is a list of tensors of shape (hidden_size).
        device (torch.device): The device to perform computations on.

    Returns:
        SteeringVectors: A dictionary containing the computed steering vectors for each layer.
            Keys are layer indices (int) and values are tensors of shape (hidden_size,).
    """
    vcd._validate_activation_list_dict(layer_dict=positive_activations)

    steering_vectors = {}
    for layer, activations in positive_activations.items():
        layer_activations = torch.stack([tensor.to(device) for tensor in activations])
        steering_vectors[layer] = torch.mean(layer_activations, dim=0)

    vcd._validate_steering_vector_dict(
        layer_dict=steering_vectors,
        expected_shape=next(iter(positive_activations.values()))[0].shape)

    return steering_vectors


def calculate_feature_expressions_kappa(
    activations: ActivationList,
    positive_activations: ActivationList,
    negative_activations: ActivationList,
    device: torch.device = torch.device("cpu")
) -> List[float]:
    """
    Calculates the kappa projection (feature expression strength) using the formula:
    proj_κ(A) = 2 * (proj_s(A) - μ_center) · ŝ
    
    Args:
        activations_list: List of activation tensors to calculate kappa for
        positive_activations: List of positive class activation tensors
        negative_activations: List of negative class activation tensors
        device: Device to perform computations on
    
    Returns:
        torch.Tensor: Kappa values for each activation
    """

    # Input validation
    if not (activations and positive_activations and negative_activations):
        raise ValueError("Input lists cannot be empty")
        
    # Move tensors to specified device
    positive_activations = [p.to(device) for p in positive_activations]
    negative_activations = [n.to(device) for n in negative_activations]
    activations = [a.to(device) for a in activations]

    # Compute steering vector as mean difference between positive and negative
    pos_mean = torch.mean(torch.stack(positive_activations), dim=0)
    neg_mean = torch.mean(torch.stack(negative_activations), dim=0)
    steering_vector = pos_mean - neg_mean
    
    # Compute center point
    mu_center = (pos_mean + neg_mean) / 2
    
    # Normalize steering vector
    s_hat = steering_vector / torch.norm(steering_vector) ** 2
    
    # Project activations onto steering vector with proj_s(A)
    projected_activations = cdr.project_activation_list_onto_steering_vector(
         activations, steering_vector)
    
    kappa = 2 * torch.sum((projected_activations - mu_center) * s_hat, dim=-1)


    # kappa = 2 * torch.sum((projected_activations - mu_center) * s_hat, dim=-1)    
    return kappa.cpu().tolist()


def calculate_feature_expressions_kappa_dict(
    activations_dict: ActivationListDict,
    positive_activations_dict: ActivationListDict,
    negative_activations_dict: ActivationListDict,
    device: torch.device = torch.device("cpu")) -> Dict[int, List[float]]:
    """
    Calculates kappa values for each layer in the activation dictionaries.
    """
    kappa_dict = {}
    for layer in activations_dict.keys():
        kappa_dict[layer] = calculate_feature_expressions_kappa(
            activations_dict[layer],
            positive_activations_dict[layer], 
            negative_activations_dict[layer],
            device
        )
    return kappa_dict