'''
Internal validation utilities for custom data types.

This module defines custom data types and validation functions for:
- Layer: NewType for layer numbers
- Activation: NewType for activation tensors
- SteeringVector: NewType for steering vectors

The following collections are defined:
- ActivationList: List of activation tensors
- ActivationListDict: Dictionary mapping layer numbers to lists of activation tensors
- SteeringVectorDict: Dictionary mapping layer numbers to steering vectors
'''
from typing import List, Dict, NewType
import torch

# Basic types
Layer = NewType('Layer', int)
Activation = NewType('Activation', torch.Tensor)
SteeringVector = NewType('SteeringVector', torch.Tensor)

# Collections
ActivationList = List[Activation]
ActivationListDict = Dict[Layer, ActivationList]
SteeringVectorDict = Dict[Layer, SteeringVector]

def _validate_layer(layer_num: int) -> Layer:
    """
    Validate and convert an integer to a Layer type.

    Args:
        layer_num: Integer to validate as layer number

    Returns:
        Layer: Validated layer number

    Raises:
        ValueError: If layer_num is negative
        TypeError: If layer_num is not an integer
    """
    if not isinstance(layer_num, int):
        raise TypeError("Layer number must be an integer")
    if layer_num < 0:
        raise ValueError("Layer number must be non-negative")
    return Layer(layer_num)

def _validate_activation(tensor: torch.Tensor) -> Activation:
    """
    Validate and convert a tensor to an Activation.

    Args:
        tensor: Tensor to validate as activation

    Returns:
        Activation: Validated activation tensor

    Raises:
        TypeError: If input is not a tensor
        ValueError: If tensor shape is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Activation must be a torch.Tensor")
    if len(tensor.shape) != 1:
        raise ValueError("Activation must be one-dimensional")
    if tensor.numel() == 0:
        raise ValueError("Tensor must not be empty")
    return Activation(tensor)

def _validate_steering_vector(tensor: torch.Tensor) -> SteeringVector:
    """
    Validate and convert a tensor to a SteeringVector.

    Args:
        tensor: Tensor to validate as steering vector

    Returns:
        SteeringVector: Validated steering vector tensor

    Raises:
        TypeError: If input is not a tensor
        ValueError: If tensor shape is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Steering vector must be a torch.Tensor")
    if len(tensor.shape) != 1:
        raise ValueError("Steering vector must be one-dimensional")
    return SteeringVector(tensor)

def _validate_activation_list(activations: List[torch.tensor]) -> ActivationList:
    """
    Validate and convert a list of tensors to an ActivationList.

    Args:
        activations: List of tensors to validate

    Returns:
        ActivationList: List of validated activations

    Raises:
        TypeError: If input is not a list or contains invalid types
        ValueError: If list is empty or tensors have inconsistent shapes
    """
    if not isinstance(activations, list):
        raise TypeError("Activations must be a list")
    if not activations:
        raise ValueError("Activation list must not be empty")
    
    validated = [_validate_activation(tensor) for tensor in activations]
    
    # Check shape consistency
    first_shape = validated[0].shape
    if not all(activation.shape == first_shape for activation in validated):
        raise ValueError("All activations must have the same shape")
    
    # Check device consistency
    first_device = validated[0].device
    if not all(activation.device == first_device for activation in validated):
        raise ValueError("All activations must be on the same device")
    
    return validated

def _validate_activation_list_dict(layer_dict: Dict[int, List[Activation]],
                                  expected_shape: torch.Size = None) -> ActivationListDict:
    """
    Validate and convert a dictionary to an ActivationListDict.

    Args:
        layer_dict: Dictionary mapping layer numbers to lists of tensors
        expected_shape: Optional expected shape for activation tensors

    Returns:
        ActivationListDict: Validated dictionary of activation lists

    Raises:
        TypeError: If input is not a dict or contains invalid types
        ValueError: If dict is empty or contains invalid data
    """
    if not isinstance(layer_dict, dict):
        raise TypeError("Input must be a dictionary")
    if not layer_dict:
        raise ValueError("Dictionary must not be empty")
    
    if expected_shape is not None:
        if not isinstance(expected_shape, torch.Size):
            raise TypeError("expected_shape must be a torch.Size")
        for layer, activations in layer_dict.items():
            for vector in activations:
                if vector.shape != expected_shape:
                    raise ValueError(f"Activation vector at layer {layer} has unexpected shape: {vector.shape}")
    
    return {
        _validate_layer(layer): _validate_activation_list(activations)
        for layer, activations in layer_dict.items()
    }

def _validate_steering_vector_dict(layer_dict: Dict[int, Activation],
                                  expected_shape: torch.Size = None) -> SteeringVectorDict:
    """
    Validate and convert a dictionary to a SteeringVectorDict.

    Args:
        layer_dict: Dictionary mapping layer numbers to steering vectors
        expected_shape: Optional expected shape for steering vectors

    Returns:
        SteeringVectorDict: Validated dictionary of steering vectors

    Raises:
        TypeError: If input is not a dict or contains invalid types
        ValueError: If dict is empty or contains invalid data
    """
    if not isinstance(layer_dict, dict):
        raise TypeError("Input must be a dictionary")
    if not layer_dict:
        raise ValueError("Dictionary must not be empty")
    if expected_shape is not None:
        if not isinstance(expected_shape, torch.Size):
            raise TypeError("expected_shape must be a torch.Size")
        for layer, vector in layer_dict.items():
            if vector.shape != expected_shape:
                raise ValueError(f"Steering vector at layer {layer} has unexpected shape: {vector.shape}")
    
    return {
        _validate_layer(layer): _validate_steering_vector(vector)
        for layer, vector in layer_dict.items()
    }

def _validate_activation_list_compatibility(
        activations1: ActivationList, activations2: ActivationList) -> None:
    """
    Validate that two activation lists are compatible for operations
    Lists are compatible if they have the same length, shape, and device

    Args:
        activations1: First list of activations
        activations2: Second list of activations

    Raises:
        ValueError: If lists are incompatible (different lengths, shapes, or devices)
        TypeError: If inputs are not ActivationList type
    """
    _validate_activation_list(activations1)
    _validate_activation_list(activations2)

    # Validate list lengths match
    if len(activations1) != len(activations2):
        raise ValueError(
            f"Length mismatch: "
            f"first list has length {len(activations1)}, "
            f"second list has length {len(activations2)}"
        )

    # Validate shape and device match
    if activations1[0].shape != activations2[0].shape:
        raise ValueError(
            f"Shape mismatch: activations have different shapes: {activations1[0].shape} and {activations2[0].shape}")
    if activations1[0].device != activations2[0].device:
        raise ValueError(
            f"Device mismatch: activations are on different devices: {activations1[0].device} and {activations2[0].device}")
    
def _validate_activation_list_to_steering_vector_compatibility(
        activations: ActivationList, steering_vector: SteeringVector) -> None:
    """
    Validate that an activation list and steering vector are compatible for operations.
    
    Args:
        activations: List of activations
        steering_vector: Steering vector
    
    Raises:
        ValueError: If activations and steering vector are incompatible
        TypeError: If inputs are not ActivationList or SteeringVector type
    """
    _validate_activation_list(activations)
    _validate_steering_vector(steering_vector)

    # Validate shape and device match
    if activations[0].shape != steering_vector.shape:
        raise ValueError("Shape mismatch: activations and steering vector have different shapes:"
                            f" {activations[0].shape} and {steering_vector.shape}")
    if activations[0].device != steering_vector.device:
        raise ValueError("Device mismatch: activations and steering vector are on different devices:"
                            f" {activations[0].device} and {steering_vector.device}")
    
def _validate_activation_list_dict_to_steering_vector_dict_compatibility(
        activations: ActivationListDict, steering_vectors: SteeringVectorDict) -> None:
    """
    Validate that activation and steering vector dictionaries are compatible for operations.

    Args:
        activations: Dictionary of activation lists
        steering_vectors: Dictionary of steering vectors
    
    Raises:
        ValueError: If activations and steering vectors are incompatible
        TypeError: If inputs are not ActivationListDict or SteeringVectorDict type
    """

    _validate_activation_list_dict(activations)
    _validate_steering_vector_dict(steering_vectors)

    # Validate layer consistency
    if activations.keys() != steering_vectors.keys():
        raise ValueError("Layer numbers must match between activations and steering vectors")
    
    for layer in activations:
       _validate_activation_list_to_steering_vector_compatibility(
            activations[layer], steering_vectors[layer]
        )

def _validate_activation_list_dict_to_activation_list_dict_compatibility(
        activations1: ActivationListDict, activations2: ActivationListDict) -> None:
    """
    Validate that two dictionaries of activation lists are compatible for operations.
    Dictionaries are compatible if they have the same keys and corresponding lists are compatible.

    Args:
        activations1: First dictionary of activation lists
        activations2: Second dictionary of activation lists

    Raises:
        ValueError: If dictionaries are incompatible (different keys or incompatible lists)
        TypeError: If inputs are not ActivationListDict type
    """

    _validate_activation_list_dict(activations1)
    _validate_activation_list_dict(activations2)

    # Validate layer consistency
    if activations1.keys() != activations2.keys():
        raise ValueError("Layer numbers must match between dictionaries")
    
    for layer in activations1:
        _validate_activation_list_compatibility(
            activations1[layer], activations2[layer]
        )


