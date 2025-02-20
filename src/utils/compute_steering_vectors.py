'''
This module contains functions to compute steering vectors from neural network activations.

The module provides three types of steering vector computation:
1. Contrastive: Computed as the mean difference between positive and negative activations
2. Non-contrastive: Computed from positive activations only
3. In-Context Vectors (ICV): Computed using PCA on the difference between positive and negative activations
'''
from typing import List, Dict
import math
import torch
import torch.nn as nn
import src.utils._validate_custom_datatypes as vcd
from src.utils._validate_custom_datatypes import (
    SteeringVector,
    ActivationList,
    ActivationListDict,
    SteeringVectorDict,
)
import src.utils.compute_dimensionality_reduction as cdr
from src.utils.pca import PCA

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

def compute_in_context_vector(
        positive_activations: ActivationListDict, negative_activations: ActivationListDict,
        device: torch.device = torch.device("cpu"), n_components: int = 1) -> SteeringVectorDict:
    """
    This function calculates the In-Context Vector (ICV) using PCA on the difference 
    between positive and negative activations for each layer.

    Args:
        positive_activations (ActivacomptionListDict): A dictionary containing positive activations for
            each layer. Each value is a list of tensors of shape (hidden_size).
        negative_activations (ActivationListDict): A dictionary containing negative activations for
            each layer. Each value is a list of tensors of shape (hidden_size).
        device (torch.device): The device to perform computations on.
        n_components (int): Number of principal components to compute. 
            Currently only n_components=1 is supported.

    Returns:
        SteeringVectorDict: A dictionary containing the computed ICVs for each layer.
            Keys are layer indices (int) and values are tensors of shape (hidden_size).
            Note: Although PCA is used, only the first principal component is returned
            as a 1D vector to maintain compatibility with other steering vectors.
    """
    if n_components != 1:
        raise NotImplementedError("Only n_components=1 is currently supported")
    
    contrastive_activations_dict_list = compute_contrastive_activations(
        positive_activations, negative_activations, device)

    in_context_vectors = {}
    for layer, contrastive_activations_list in contrastive_activations_dict_list.items():        
        contrastive_activations = torch.stack(contrastive_activations_list).to(device)
        
        # Perform PCA
        pca = PCA(device=device, n_components=n_components)
        pca.fit(contrastive_activations)
        in_context_vectors[layer] = pca.components_[0]

    vcd._validate_steering_vector_dict(
        layer_dict=in_context_vectors,
        expected_shape=next(iter(positive_activations.values()))[0].shape)

    return in_context_vectors

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

def compute_mimic_steering_vector(
        positive_activations: ActivationListDict,
        negative_activations: ActivationListDict,
        eps: float = 0.0,
        low_data_eps: float = 1.0,
        halfway_point: float = 0.5,
        low_data_regime_multiplier: float = 1.5,
        device: torch.device = torch.device("cpu")) -> SteeringVectorDict:
    """
    Implements the MiMiC method for computing steering vectors based on mean and covariance matching.
    
    Args:
        positive_activations (ActivationListDict): Dictionary of positive activations by layer
        negative_activations (ActivationListDict): Dictionary of negative activations by layer
        eps (float): Base regularization strength
        low_data_eps (float): Stronger regularization for low-data regime
        halfway_point (float): Fraction of dimension at which regularization is halfway
        low_data_regime_multiplier (float): Multiplier for low data regime threshold
        device (torch.device): Device to perform computations on

    Returns:
        SteeringVectorDict: Dictionary containing steering vectors for each layer
    """
    vcd._validate_activation_list_dict_to_activation_list_dict_compatibility(
        activations1=positive_activations, activations2=negative_activations)
    
    steering_vectors = {}
    
    for layer in positive_activations:
        # Stack activations into matrices
        pos_acts = torch.stack(positive_activations[layer]).to(device)
        neg_acts = torch.stack(negative_activations[layer]).to(device)
        input_dim = pos_acts.shape[1]
        
        # Compute means
        mu_0 = neg_acts.mean(dim=0)
        mu_1 = pos_acts.mean(dim=0)
        
        # Determine regularization strength
        n_pairs = len(pos_acts)
        n_pairs_reg_off = input_dim * low_data_regime_multiplier
        if n_pairs < n_pairs_reg_off:
            rate = -math.log(0.5) / (halfway_point * n_pairs_reg_off)
            t = n_pairs / n_pairs_reg_off
            decay = math.exp(-rate * t)
            reg_strength = eps + (low_data_eps - eps) * decay
        else:
            reg_strength = eps
            
        # Compute covariances with regularization
        sigma_0 = torch.cov(neg_acts.T) + reg_strength * torch.eye(input_dim, device=device)
        sigma_1 = torch.cov(pos_acts.T) + reg_strength * torch.eye(input_dim, device=device)
        
        # Compute transformation matrix W using SVD
        U, S, Vt = torch.linalg.svd(sigma_0)
        sigma_0_sqrt = U @ torch.diag(torch.sqrt(S)) @ Vt
        sigma_0_sqrt_pinv = U @ torch.diag(1.0 / torch.sqrt(S)) @ Vt
        
        inner_term = sigma_0_sqrt @ sigma_1 @ sigma_0_sqrt.T
        U_inner, S_inner, Vt_inner = torch.linalg.svd(inner_term)
        inner_sqrt = U_inner @ torch.diag(torch.sqrt(S_inner)) @ Vt_inner
        
        W = sigma_0_sqrt_pinv @ inner_sqrt @ sigma_0_sqrt_pinv.T
        b = mu_1 - W @ mu_0
        
        # Store the steering vector (transformation matrix and bias combined)
        steering_vectors[layer] = (W, b)
    
    return steering_vectors

def compute_paired_icv_steering_vector(
        positive_activations: ActivationListDict,
        negative_activations: ActivationListDict,
        multiplier: float = 1.0,
        device: torch.device = torch.device("cpu")) -> SteeringVectorDict:
    """
    Implements the Paired In-Context Vector (ICV) method for computing steering vectors.
    
    Args:
        positive_activations (ActivationListDict): Dictionary of positive activations by layer
        negative_activations (ActivationListDict): Dictionary of negative activations by layer
        multiplier (float): Scaling factor for the ICV (lambda in the paper)
        device (torch.device): Device to perform computations on

    Returns:
        SteeringVectorDict: Dictionary containing steering vectors for each layer
    """
    vcd._validate_activation_list_dict_to_activation_list_dict_compatibility(
        activations1=positive_activations, activations2=negative_activations)
    
    steering_vectors = {}
    
    for layer in positive_activations:
        pos_acts = torch.stack(positive_activations[layer]).to(device)
        neg_acts = torch.stack(negative_activations[layer]).to(device)
        
        # Compute activation differences
        delta_h = pos_acts - neg_acts
        
        # Compute first principal component
        U, _, _ = torch.svd(delta_h.t())
        icv = U[:, 0] * multiplier
        
        steering_vectors[layer] = icv
    
    vcd._validate_steering_vector_dict(
        layer_dict=steering_vectors,
        expected_shape=next(iter(positive_activations.values()))[0].shape)
    
    return steering_vectors

def apply_mimic_steering(
        activations: ActivationListDict,
        steering_vectors: SteeringVectorDict,
        device: torch.device = torch.device("cpu")) -> ActivationListDict:
    """
    Applies MiMiC steering transformation to activations.
    
    Args:
        activations (ActivationListDict): Dictionary of activations by layer
        steering_vectors (SteeringVectorDict): Dictionary of (W, b) tuples by layer
        device (torch.device): Device to perform computations on
        
    Returns:
        ActivationListDict: Transformed activations
    """
    transformed_activations = {}
    
    for layer in activations:
        W, b = steering_vectors[layer]
        layer_acts = [act.to(device) for act in activations[layer]]
        transformed = [(W @ act.unsqueeze(-1)).squeeze(-1) + b for act in layer_acts]
        transformed_activations[layer] = transformed
    
    return transformed_activations

def apply_paired_icv_steering(
        activations: ActivationListDict,
        steering_vectors: SteeringVectorDict,
        device: torch.device = torch.device("cpu")) -> ActivationListDict:
    """
    Applies Paired ICV steering transformation to activations.
    
    Args:
        activations (ActivationListDict): Dictionary of activations by layer
        steering_vectors (SteeringVectorDict): Dictionary of ICVs by layer
        device (torch.device): Device to perform computations on
        
    Returns:
        ActivationListDict: Transformed activations
    """
    transformed_activations = {}
    
    for layer in activations:
        icv = steering_vectors[layer]
        layer_acts = [act.to(device) for act in activations[layer]]
        
        transformed = []
        for act in layer_acts:
            # Shift activation by ICV
            act_shifted = act + icv
            
            # Normalize to preserve original norm
            act_norm = torch.norm(act)
            act_shifted_norm = torch.norm(act_shifted)
            transformed.append(act_shifted * (act_norm / act_shifted_norm))
            
        transformed_activations[layer] = transformed
    
    return transformed_activations

class MeanCovarianceMatchingSteering(nn.Module):
    """MiMiC method from the paper: https://arxiv.org/abs/2402.09631"""
    def __init__(self, input_dim, eps=0.0, low_data_eps=1.0, halfway_point=0.5, low_data_regime_multiplier=1.5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps  # regular regularization strength
        self.low_data_eps = low_data_eps  # stronger regularization for low-data regime
        self.halfway_point = halfway_point  # fraction of dim at which regularization is halfway between low_data_eps and eps
        self.low_data_regime_multiplier = low_data_regime_multiplier
        self.register_buffer('W', None)
        self.register_buffer('b', None)

    def interpolate_eps(self, n_pairs) -> float:
        """
        Interpolate eps based on number of training data pairs.
        Reaches halfway point when n_pairs = halfway_point * n_pairs_reg_off
        """
        n_pairs_reg_off = self.input_dim * self.low_data_regime_multiplier  # num train datapoints when we fully switch away from low_data_eps
        if n_pairs >= n_pairs_reg_off:
            return self.eps
        
        rate = -math.log(0.5) / (self.halfway_point * n_pairs_reg_off)
        t = n_pairs / n_pairs_reg_off
        decay = math.exp(-rate * t)
        return self.eps + (self.low_data_eps - self.eps) * decay

    def compute_steering_params(self, activations_absent, activations_present) -> None:
        # Compute means
        mu_0 = activations_absent.mean(dim=0)
        mu_1 = activations_present.mean(dim=0)

        # Get number of pairs and dimension
        n_pairs = len(activations_absent)

        # Determine interpolated regularization strength
        reg_strength = self.interpolate_eps(n_pairs)

        # Compute covariances
        sigma_0 = torch.cov(activations_absent.T)
        sigma_1 = torch.cov(activations_present.T)

        # Ensure covariance matrices are positive definite with appropriate regularization
        sigma_0 += reg_strength * torch.eye(sigma_0.shape[0], device=sigma_0.device)
        sigma_1 += reg_strength * torch.eye(sigma_1.shape[0], device=sigma_1.device)

        # Compute W using SVD
        U, S, Vt = torch.linalg.svd(sigma_0)
        sigma_0_sqrt = U @ torch.diag(torch.sqrt(S)) @ Vt
        sigma_0_sqrt_pinv = U @ torch.diag(1.0 / torch.sqrt(S)) @ Vt

        inner_term = sigma_0_sqrt @ sigma_1 @ sigma_0_sqrt.T
        U_inner, S_inner, Vt_inner = torch.linalg.svd(inner_term)
        inner_sqrt = U_inner @ torch.diag(torch.sqrt(S_inner)) @ Vt_inner

        self.W = sigma_0_sqrt_pinv @ inner_sqrt @ sigma_0_sqrt_pinv.T

        # Compute b
        self.b = mu_1 - self.W @ mu_0

    def forward(self, x):
        if self.W is None or self.b is None:
            raise ValueError("Steering parameters have not been computed. Call compute_steering_params first.")
        
        original_shape = x.shape
        assert original_shape[-1] == self.input_dim, f"Last dimension of input must be {self.input_dim}, got {original_shape[-1]}"
        
        x_flat = x.view(-1, self.input_dim)  # Reshape to 2D: [*, dim] where * is the product of all other dimensions
        transformed = (self.W @ x_flat.T).T + self.b  # Apply steering
        
        return transformed.view(original_shape)  # Reshape back to original shape

    def get_full_transformation_matrix(self):
        return self.W
    

class PairedICV(nn.Module):
    """In-Context Vector (ICV) for paired datapoints https://arxiv.org/abs/2311.06668"""
    def __init__(self, input_dim, multiplier=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.multiplier = multiplier  # Lambda in the paper
        self.register_buffer('icv', None)

    def compute_steering_params(self, acts_x, acts_y):
        delta_h = acts_y - acts_x  # shape: [num_paired_datapoints, dim]       
        U, _, _ = torch.svd(delta_h.t())
        self.icv = U[:, 0]  # First principal direction

    def forward(self, x):
        if self.icv is None:
            raise ValueError("ICV has not been computed. Call compute_steering_params first.")
        
        x_shifted = x + self.multiplier * self.icv.unsqueeze(0)
        
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_shifted_norm = torch.norm(x_shifted, dim=-1, keepdim=True)
        
        return x_shifted * (x_norm / x_shifted_norm)  # Normalize the to preserve the norm of the input
