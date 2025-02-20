'''
pca.py implements the PCA module in PyTorch for dimensionality reduction. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def svd_flip(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adjusts the signs of singular vectors for consistency.

    Args:
        u (torch.Tensor): Left singular vectors.
        v (torch.Tensor): Right singular vectors.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Adjusted u and v tensors.
    """
    max_abs_cols = torch.argmax(input=torch.abs(u), dim=0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, device: torch.device, n_components: int = 1) -> None:
        """
        Initialize the PCA module.

        Args:
            n_components (int | None): Number of components to keep. If None, keep all components.
        """
        super().__init__()
        self.n_components = n_components
        self.device = device

    def _check_input(self, X: torch.Tensor, for_fit: bool = False) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Input data must be a PyTorch tensor, but got {type(X).__name__}.")
        
        if X.dim() != 2:
            raise ValueError(f"Input data must be a 2-dimensional tensor, but got a tensor of shape {X.shape}.")
        
        if X.isnan().any():
            raise ValueError("Input data contains NaN values.")
        
        if X.isinf().any():
            raise ValueError("Input data contains infinity values.")
        
        if for_fit:
            n_samples, n_features = X.size()
            if n_samples < 2:
                raise ValueError(f"n_samples must be greater than 1, but got {n_samples}.")
            
            if n_features < 2:
                raise ValueError(f"n_features must be greater than 1, but got {n_features}.")
            
            if self.n_components < 1:
                raise ValueError(f"n_components must be >= 1, but got {self.n_components}.")
            
            if self.n_components > min(n_samples, n_features):
                raise ValueError(f"n_components must be <= min(n_samples, n_features). Got {self.n_components} instead of at most {min(n_samples, n_features)}.")
        
        return X.float()

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> 'PCA':
        """
        Fit the PCA model to the data.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            PCA: The fitted PCA object.
        """
        X = self._check_input(X, for_fit=True)

        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_  # center the data
        if self.device != torch.device("cpu"):
            Z = Z.to(torch.device("cpu"))
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        if self.device != torch.device("cpu"):
            U = U.to(self.device)
            S = S.to(self.device)
            Vh = Vh.to(self.device)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)

        self.register_buffer("components_", Vt[:self.n_components])
        self.register_buffer("singular_values_", S[:self.n_components])

        total_var = torch.sum(S ** 2)
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        self.register_buffer("explained_variance_", explained_variance[:self.n_components])
        self.register_buffer("explained_variance_ratio_", (S ** 2) / total_var)
        

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply dimensionality reduction to X.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Transformed data of shape (n_samples, n_components).
        """
        assert hasattr(self, "components_") and hasattr(self, "mean_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Transform data back to its original space.

        Args:
            Y (torch.Tensor): Data in reduced space of shape (n_samples, n_components).

        Returns:
            torch.Tensor: Data in original space of shape (n_samples, n_features).
        """
        assert hasattr(self, "components_") and hasattr(self, "mean_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PCA module.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Transformed data of shape (n_samples, n_components).
        """
        return self.transform(X)