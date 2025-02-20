'''
compute_dimensionality_reduction.py
TODO: Implement the following functions for dimensionality reduction:
# PCA
# t-Distributed Stochastic Neighbor Embedding (t-SNE)
# Uniform Manifold Approximation and Projection (UMAP)

'''
from typing import List, NewType, Dict, Optional, Tuple
import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import src.utils._validate_custom_datatypes as vcd
from src.utils._validate_custom_datatypes import (
    Layer,
    SteeringVector,
    SteeringVectorDict,
    Activation,
    ActivationList,
    ActivationListDict,
)
LDAProjection = NewType('LDAProjection', torch.Tensor)
LDATransformation = NewType('LDATransformation', torch.Tensor)

def project_activation_list_onto_steering_vector(activations_list: List[torch.Tensor],
                                                 steering_vector: torch.Tensor) -> torch.Tensor:
    """
    Projects a list of activation tensors onto a steering vector using the formula:
    proj_s(A) = (A · s / ||s||^2) * s
    
    Args:
        activations_list: List of activation tensors, each of shape (d_e,)
        steering_vector: Steering direction tensor of shape (d_e,)
    
    Returns:
        torch.Tensor: Stacked tensor of projections with shape (num_activations, d_e)
        
    Raises:
        ValueError: If steering_vector is zero vector
    """
    # Stack activations into single tensor
    activations = torch.stack(activations_list)
    
    # Calculate squared norm of steering vector (||s||^2)
    s_norm_squared = torch.dot(steering_vector, steering_vector)
    if s_norm_squared == 0:
        raise ValueError("Steering vector cannot be zero vector")
        
    # Calculate projection coefficients for all activations at once
    coeffs = torch.matmul(activations, steering_vector) / s_norm_squared
    
    # Return projections
    return coeffs.unsqueeze(-1) * steering_vector

def scalar_projection_of_activation_list_onto_steering_vector(
        activations_list: List[torch.Tensor], steering_vector: torch.Tensor) -> np.ndarray:
    """
    Compute scalar projection of activations onto normalized steering vector.
    Formula: proj_scalar(A) = A·ŝ where ŝ = s/||s||
    
    Args:
        activations_list: List of activation tensors
        steering_vector: Steering vector to project onto
    
    Returns:
        tensor of scalar projection values
    """
    # Normalize steering vector (ŝ = s/||s||)
    s_hat = steering_vector / torch.norm(steering_vector)
    
    # Stack activations
    activations = torch.stack(activations_list)
    
    # Compute scalar projections (A·ŝ)
    projections = torch.matmul(activations, s_hat)
    
    return projections.cpu().numpy()

def compute_binary_pca_projection(
    positive_activations: ActivationList,
    negative_activations: ActivationList,
    n_components: int = 2,
    device: Optional[torch.device] = None
) -> Tuple[LDAProjection, LDATransformation, Dict[str, float]]:
    """
    Project positive and negative activations into lower-dimensional space using Principal Component Analysis.
    While PCA is unsupervised, this function maintains the two-class structure for consistency and evaluation.
    
    Args:
        positive_activations: List of activation vectors for positive class
        negative_activations: List of activation vectors for negative class
        n_components: Number of PCA components to compute (default=2)
        device: Target device for computation. If None, uses the device of input tensors.
        
    Returns:
        Tuple containing:
        - Projected activations (n_samples × n_components): Lower-dimensional projection of all input vectors
        - PCA transformation matrix (n_features × n_components): The principal components
        - Dictionary of quality metrics:
            - 'explained_variance_ratio': List of variance explained by each component
            - 'cumulative_variance_ratio': Cumulative sum of explained variance ratios
            - 'separation_scores': Class separation scores for each component
            - 'condition_number': Condition number of the covariance matrix
        
    Raises:
        ValueError: If input dimensions don't match, if PCA fails, or if inputs are empty
        TypeError: If inputs are not valid ActivationLists or not floating point
        RuntimeWarning: If numerical instability is detected
    """
    from sklearn.decomposition import PCA
    
    # Validate inputs
    vcd.validate_activation_list_compatibility(
        activations1=positive_activations, 
        activations2=negative_activations
    )
    
    if len(positive_activations) < 1 or len(negative_activations) < 1:
        raise ValueError("Each class must have at least 1 sample for PCA")
    
    # Set device if not provided
    if device is None:
        device = positive_activations[0].device
        
    # Stack and prepare data
    stacked_pos = torch.stack(positive_activations)
    stacked_neg = torch.stack(negative_activations)
    X = torch.cat([stacked_pos, stacked_neg], dim=0)
    
    # Store labels for later evaluation
    labels = torch.cat([
        torch.ones(len(positive_activations)),
        torch.zeros(len(negative_activations))
    ]).to(device)
    
    # Check for zero variance features
    total_var = torch.var(X, dim=0)
    if (total_var < 1e-10).any():
        raise ValueError("Near-zero variance detected in some features")
    
    # Convert to numpy for sklearn
    X_numpy = X.cpu().numpy()
    
    # Check condition number for numerical stability
    cov_matrix = np.cov(X_numpy.T)
    condition_number = np.linalg.cond(cov_matrix)
    if condition_number > 1e6:  # threshold for poor conditioning
        import warnings
        warnings.warn(f"High condition number ({condition_number:.2e}) indicates potential numerical instability")
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    try:
        X_transformed_numpy = pca.fit_transform(X_numpy)
    except Exception as e:
        raise ValueError(f"PCA failed: {str(e)}")
    
    # Convert back to torch tensors
    projection = torch.from_numpy(X_transformed_numpy).to(device)
    transform_matrix = torch.from_numpy(pca.components_.T).to(device)
    
    # Compute quality metrics
    quality_metrics = {}
    
    # 1. Explained variance ratios
    quality_metrics['explained_variance_ratio'] = [
        float(ratio) for ratio in pca.explained_variance_ratio_
    ]
    quality_metrics['cumulative_variance_ratio'] = float(
        np.sum(pca.explained_variance_ratio_)
    )
    
    # 2. Separation scores for each component
    separation_scores = []
    for component_idx in range(n_components):
        pos_proj = projection[:len(positive_activations), component_idx]
        neg_proj = projection[len(positive_activations):, component_idx]
        
        pos_mean, pos_std = torch.mean(pos_proj), torch.std(pos_proj)
        neg_mean, neg_std = torch.mean(neg_proj), torch.std(neg_proj)
        
        # Compute separation score (similar to LDA implementation)
        separation_score = abs(pos_mean - neg_mean) / (pos_std + neg_std)
        separation_scores.append(float(separation_score))
    
    quality_metrics['separation_scores'] = separation_scores
    quality_metrics['condition_number'] = float(condition_number)
    
    return LDAProjection(projection), LDATransformation(transform_matrix), quality_metrics


def compute_binary_lda_projection(
    positive_activations: ActivationList,
    negative_activations: ActivationList,
    device: Optional[torch.device] = None
) -> Tuple[LDAProjection, LDATransformation, Dict[str, float]]:
    """
    Project positive and negative activations into 1D space using Linear Discriminant Analysis.
    This finds the direction that best separates the two classes.
    
    Args:
        positive_activations: List of activation vectors for positive class
        negative_activations: List of activation vectors for negative class
        device: Target device for computation. If None, uses the device of input tensors.
        
    Returns:
        Tuple containing:
        - Projected activations (n_samples × 1): 1D projection of all input vectors
        - LDA transformation matrix (n_features × 1): The direction that best separates classes
        - Dictionary of quality metrics:
            - 'explained_variance_ratio': Proportion of variance explained
            - 'separation_score': Difference between class means divided by sum of standard deviations
            - 'classification_accuracy': Accuracy of LDA on training data
        
    Raises:
        ValueError: If input dimensions don't match, if LDA fails, or if inputs are empty
        TypeError: If inputs are not valid ActivationLists or not floating point
        RuntimeWarning: If numerical instability is detected
    """
    vcd.validate_activation_list_compatibility(activations1=positive_activations, 
                                               activations2=negative_activations)
    
    if len(positive_activations) < 2 or len(negative_activations) < 2:
        raise ValueError("Each class must have at least 2 samples for LDA")
            
    # Check for zero variance features
    stacked_pos = torch.stack(positive_activations)
    stacked_neg = torch.stack(negative_activations)
    total_var = torch.var(torch.cat([stacked_pos, stacked_neg], dim=0), dim=0)
    if (total_var < 1e-10).any():
        raise ValueError("Near-zero variance detected in some features")

    # Set device if not provided
    if device is None:
        device = positive_activations[0].device
        
    # Prepare data
    X = np.vstack([stacked_pos.cpu().numpy(), stacked_neg.cpu().numpy()])
    
    y = np.concatenate([np.ones(len(positive_activations)), np.zeros(len(negative_activations))])
    
    # Check condition number for numerical stability
    cov_matrix = np.cov(X.T)
    condition_number = np.linalg.cond(cov_matrix)
    if condition_number > 1e6:  # threshold for poor conditioning
        import warnings
        warnings.warn(f"High condition number ({condition_number:.2e}) indicates potential numerical instability")
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    try:
        X_transformed = lda.fit_transform(X, y)
    except Exception as e:
        raise ValueError(f"LDA failed: {str(e)}")
    
    # Validate output dimensions
    if X_transformed.shape[1] != 1:
        raise ValueError(f"LDA output has unexpected dimensions. Expected 1, got {X_transformed.shape[1]}")
    if lda.scalings_.shape[1] != 1:
        raise ValueError(f"LDA transformation matrix has unexpected dimensions. Expected 1 component")
        
    # Convert back to torch
    projection = torch.from_numpy(X_transformed).to(device)
    transform_matrix = torch.from_numpy(lda.scalings_).to(device)
    
    # Compute quality metrics
    # 1. Separation score
    pos_proj = projection[:len(positive_activations)]
    neg_proj = projection[len(positive_activations):]
    pos_mean, pos_std = torch.mean(pos_proj), torch.std(pos_proj)
    neg_mean, neg_std = torch.mean(neg_proj), torch.std(neg_proj)
    separation_score = abs(pos_mean - neg_mean) / (pos_std + neg_std)
    
    # 2. Classification accuracy
    predictions = lda.predict(X)
    accuracy = (predictions == y).mean()
    
    # 3. Explained variance ratio
    explained_variance = lda.explained_variance_ratio_[0]
    
    quality_metrics = {
        'explained_variance_ratio': float(explained_variance),
        'separation_score': float(separation_score),
        'classification_accuracy': float(accuracy)
    }
    
    return LDAProjection(projection), LDATransformation(transform_matrix), quality_metrics