import unittest
import torch
import numpy as np
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Get the utils path from the environment variable
utils_path = os.getenv('UTILS_PATH')

# Add the utils path to sys.path if it's not None
if utils_path:
    sys.path.append(utils_path)
else:
    raise ValueError("UTILS_PATH environment variable is not set")

# Now import PCA
from pca import PCA

class TestPCA(unittest.TestCase):
    def setUp(self):
        self.X = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        self.pca = PCA()  # Default n_components=1

    def test_initialization(self):
        self.assertEqual(self.pca.n_components, 1)

    def test_fit(self):
        self.pca.fit(self.X)
        self.assertTrue(hasattr(self.pca, 'mean_'))
        self.assertTrue(hasattr(self.pca, 'components_'))
        self.assertEqual(self.pca.components_.shape, (1, 3))

    def test_transform(self):
        self.pca.fit(self.X)
        transformed = self.pca.transform(self.X)
        self.assertEqual(transformed.shape, (4, 1))

    def test_fit_transform(self):
        transformed = self.pca.fit_transform(self.X)
        self.assertEqual(transformed.shape, (4, 1))

    def test_inverse_transform(self):
        transformed = self.pca.fit_transform(self.X)
        reconstructed = self.pca.inverse_transform(transformed)
        self.assertEqual(reconstructed.shape, (4, 3))
        # Note: With only 1 component, the reconstruction error might be larger
        self.assertTrue(torch.allclose(self.X, reconstructed, atol=1e-1))

    def test_forward(self):
        self.pca.fit(self.X)
        output = self.pca(self.X)
        self.assertEqual(output.shape, (4, 1))

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            self.pca.fit(np.array([1, 2, 3]))  # Not a torch.Tensor
        
        with self.assertRaises(ValueError):
            self.pca.fit(torch.tensor([1, 2, 3]))  # Not 2D
        
        with self.assertRaises(ValueError):
            self.pca.fit(torch.tensor([[1, 2], [3, float('nan')]]))  # Contains NaN
        
        with self.assertRaises(ValueError):
            self.pca.fit(torch.tensor([[1, 2], [3, float('inf')]]))  # Contains inf

    def test_zero_mean_data(self):
        X = torch.tensor([
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0]
        ])
        pca = PCA(n_components=2)
        pca.fit(X)
        self.assertTrue(torch.allclose(pca.mean_, torch.tensor([[0.0, 0.0]]), atol=1e-6))
        
        # Check orthogonality
        dot_product = torch.dot(pca.components_[0], pca.components_[1])
        self.assertTrue(torch.isclose(dot_product, torch.tensor(0.0), atol=1e-6))
        
        # Check that the components explain all the variance
        self.assertTrue(torch.allclose(torch.sum(pca.explained_variance_ratio_), torch.tensor(1.0), atol=1e-6))

    def test_custom_n_components(self):
        pca = PCA(n_components=2)
        pca.fit(self.X)
        self.assertEqual(pca.components_.shape[0], 2)

    def test_n_components_exceeding_limit(self):
        pca = PCA(n_components=5)  # Exceeds min(n_samples, n_features)
        with self.assertRaises(ValueError) as context:
            pca.fit(self.X)
        
        self.assertTrue('n_components must be <= min(n_samples, n_features)' in str(context.exception))

    def test_n_components_at_limit(self):
        pca = PCA(n_components=3)  # Equal to min(n_samples, n_features)
        pca.fit(self.X)
        self.assertEqual(pca.components_.shape[0], 3)
        
        transformed = pca.transform(self.X)
        self.assertEqual(transformed.shape, (4, 3))

    def test_reconstruction_error(self):
        X = torch.randn(100, 10)
        for n_components in range(1, 11):
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(X)
            reconstructed = pca.inverse_transform(transformed)
            error = torch.mean((X - reconstructed) ** 2)
            self.assertLess(error, 1e-5 if n_components == 10 else 1.0)

    def test_large_dimension_reduction(self):
        X = torch.randn(1000, 100)
        pca = PCA(n_components=10)
        transformed = pca.fit_transform(X)
        self.assertEqual(transformed.shape, (1000, 10))


    def test_constant_feature(self):
        X = torch.tensor([
            [1.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, 2.0, 5.0],
            [4.0, 2.0, 5.0]
        ])
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(X)
        self.assertEqual(transformed.shape, (4, 2))
        
        # The constant feature should not contribute to the principal components
        self.assertTrue(torch.allclose(pca.components_[:, 1], torch.tensor([0.0, 0.0]), atol=1e-6))

    def test_identical_samples(self):
        X = torch.tensor([[1.0, 2.0, 3.0]] * 4)
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(X)
        self.assertEqual(transformed.shape, (4, 2))
        
        # All transformed values should be zero
        self.assertTrue(torch.allclose(transformed, torch.zeros_like(transformed), atol=1e-6))

    def test_orthogonality_of_components(self):
        pca = PCA(n_components=2)
        pca.fit(self.X)
        dot_product = torch.dot(pca.components_[0], pca.components_[1])
        self.assertTrue(torch.isclose(dot_product, torch.tensor(0.0), atol=1e-6))

    def test_variance_explanation(self):
        X = torch.randn(100, 5)
        pca = PCA(n_components=5)
        pca.fit(X)
        
        # Check that explained variance ratios sum to 1
        self.assertTrue(torch.isclose(torch.sum(pca.explained_variance_ratio_), torch.tensor(1.0), atol=1e-6))
        
        # Check that explained variance ratios are in descending order
        self.assertTrue(torch.all(pca.explained_variance_ratio_[:-1] >= pca.explained_variance_ratio_[1:]))

if __name__ == '__main__':
    unittest.main()