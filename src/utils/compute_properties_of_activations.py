from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
import src.utils._validate_custom_datatypes as vcd
from src.utils._validate_custom_datatypes import (
    Layer,
    SteeringVector,
    SteeringVectorDict,
    ActivationList,
    ActivationListDict,
)

def compute_pairwise_cosine_similarity(
        activation_list: ActivationList, device: Optional[torch.device] = None) -> List[float]:
    """
    Compute pairwise cosine similarity between all activation vectors.
    Returns a list of unique similarities (no duplicates).
    
    Args:
        activations: List of 1D torch tensors
        device: Target device for computation. If None, uses the device of the first activation

    Returns:
        List[float]: List of unique pairwise cosine similarities

    Raises:
        TypeError: If activations is not a valid ActivationList
        ValueError: If activations list is empty, contains invalid tensors or activations are on different devices.
    """
    activation_list = vcd._validate_activation_list(activation_list)
    if device is None:
        device = activation_list[0].device

    vectors_stacked = torch.stack([activation.to(device) for activation in activation_list])
    vectors_normalized = F.normalize(vectors_stacked, p=2, dim=1)
    similarity_matrix = torch.mm(vectors_normalized, vectors_normalized.T)

    # Get upper triangular part of matrix (excluding diagonal)
    mask = torch.ones_like(similarity_matrix, dtype=torch.bool).triu(diagonal=1)
    similarities = similarity_matrix[mask].tolist()
    return similarities

def compute_pairwise_cosine_similarity_dict(
        activation_list_dict: ActivationListDict, 
        device: Optional[torch.device] = None) -> Dict[Layer, List[float]]:
    
    """
    Compute pairwise cosine similarity between all activation vectors within each layer.

    Args:
        activations: Dictionary mapping layers to lists of 1D torch tensors
        device: Target device for computation. If None, uses the device of the first activation
    
    Returns:
        Dict[Layer, List[float]]: Dictionary mapping layers to lists of unique pairwise similarities
    """
    vcd._validate_activation_list_dict(activation_list_dict)
    if device is None:
        device = next(iter(activation_list_dict.values()))[0].device
    
    similarities_dict = {}
    for layer, layer_activations in activation_list_dict.items():
        similarities_dict[layer] = compute_pairwise_cosine_similarity(layer_activations, device)
    
    return similarities_dict

def compute_many_against_one_cosine_similarity(
              activations: ActivationList, steering_vector: SteeringVector, 
              device: Optional[torch.device] = None) -> List[float]:
    """
    Compute cosine similarity between each activation vector and the steering vector.
    
    Args:
        activations: List of 1D torch tensors
        steering_vector: 1D torch tensor
        device: Target device for computation. If None, uses the device of the steering vector
        
    Returns:
        List[float]: List of cosine similarities with steering vector

    Raises:
        TypeError: If inputs are not valid ActivationList and SteeringVector
        ValueError: If inputs are incompatible or invalid
    """
    vcd._validate_activation_list_to_steering_vector_compatibility(activations, steering_vector)

    if device is None:
        device = steering_vector.device

    vectors_stacked = torch.stack([activation.to(device) for activation in activations])
    steering_vector = steering_vector.to(device)

    vectors_normalized = F.normalize(vectors_stacked, p=2, dim=1)
    steering_normalized = F.normalize(steering_vector, p=2, dim=0).unsqueeze(0)
    
    cosine_similarities = torch.mm(vectors_normalized, steering_normalized.T).squeeze()
    
    return cosine_similarities.tolist()

def compute_many_against_one_cosine_similarity_dict(
        activation_list_dict: ActivationListDict, steering_vector_dict: SteeringVectorDict,
        device: Optional[torch.device] = None) -> Dict[Layer, List[float]]:
    
    """
    Compute cosine similarity between each activation vector and the steering vector for each layer.

    Args:
        activation_list_dict: Dictionary mapping layers to lists of 1D torch tensors
        steering_vector_dict: Dictionary mapping layers to steering vectors
        device: Target device for computation. If None, uses the device of the first steering vector
    
    Returns:
        Dict[Layer, List[float]]: Dictionary mapping layers to lists of cosine similarities
    """
    vcd._validate_activation_list_dict_to_steering_vector_dict_compatibility(
        activation_list_dict, steering_vector_dict)
    
    if device is None:
        device = next(iter(activation_list_dict.values()))[0].device

    similarities_dict = {}
    for layer in activation_list_dict:
        similarities_dict[layer] = compute_many_against_one_cosine_similarity(
            activation_list_dict[layer], steering_vector_dict[layer], device)
    
    return similarities_dict

def compute_l2_norms(activations: ActivationList,
                     device: Optional[torch.device] = None) -> List[float]:
    """
    Compute L2 norm for each vector in the list.
    
    Args:
        activations: List of 1D torch tensor
        device: Target device for computation. If None, uses the device of the first activation
        
    Returns:
        List[float]: List of L2 norms
    """
    activations = vcd._validate_activation_list(activations)

    if device is None:
        device = activations[0].device

    vectors_stacked = torch.stack([activation.to(device) for activation in activations])
    norms = torch.norm(vectors_stacked, p=2, dim=1)
    
    return norms.tolist()

def compute_l2_norms_dict(activation_list_dict: ActivationListDict,
                          device: Optional[torch.device] = None) -> Dict[Layer, List[float]]:
        """
        Compute L2 norm for each vector in each layer.
        
        Args:
            activation_list_dict: Dictionary mapping layers to lists of 1D torch tensors
            device: Target device for computation. If None, uses the device of the first activation
            
        Returns:
            Dict[Layer, List[float]]: Dictionary mapping layers to lists of L2 norms
        """
        vcd._validate_activation_list_dict(activation_list_dict)
    
        if device is None:
            device = next(iter(activation_list_dict.values()))[0].device
    
        norms_dict = {}
        for layer, layer_activations in activation_list_dict.items():
            vectors_stacked = torch.stack([activation.to(device) for activation in layer_activations])
            norms = torch.norm(vectors_stacked, p=2, dim=1)
            norms_dict[layer] = norms.tolist()
    
        return norms_dict
