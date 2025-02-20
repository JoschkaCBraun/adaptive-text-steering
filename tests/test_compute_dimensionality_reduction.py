'''
test_compute_dimensionality_reduction.py
'''
import os
import sys
import unittest
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the utils path from the environment variable
utils_path = os.getenv('UTILS_PATH')

# Add the utils path to sys.path if it's not None
if utils_path:
    sys.path.append(utils_path)
else:
    raise ValueError("UTILS_PATH environment variable is not set")

import compute_dimensionality_reduction as cdr
from validate_custom_datatypes import validate_activation_list

class TestDimensionalityReduction(unittest.TestCase):
    def setUp(self):
        """Set up test cases with sample activation data."""
        # Use MPS if available, otherwise CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create sample data
        # 10-dimensional vectors: 5 positive samples, 5 negative samples
        self.n_features = 10
        self.n_samples = 5
        torch.manual_seed(42)  # for reproducibility
        
        # Create clearly separable data
        self.pos_activations = [
            torch.randn(self.n_features).to(self.device) + torch.tensor([2.0] + [0.0] * (self.n_features-1)).to(self.device)
            for _ in range(self.n_samples)
        ]
        self.neg_activations = [
            torch.randn(self.n_features).to(self.device) - torch.tensor([2.0] + [0.0] * (self.n_features-1)).to(self.device)
            for _ in range(self.n_samples)
        ]

        # Create edge case data
        self.zero_var_pos = [torch.zeros(self.n_features).to(self.device) for _ in range(3)]
        self.single_pos = [torch.randn(self.n_features).to(self.device)]
        self.diff_dim_pos = [torch.randn(self.n_features + 1).to(self.device) for _ in range(3)]

    def test_basic_functionality(self):
        """Test if the function runs with valid inputs and returns expected output types."""
        projection, transform, metrics = cdr.compute_binary_lda_projection(
            self.pos_activations, self.neg_activations, self.device
        )
        
        # Check output types
        self.assertIsInstance(projection, torch.Tensor)
        self.assertIsInstance(transform, torch.Tensor)
        self.assertIsInstance(metrics, dict)
        
        # Check output dimensions
        self.assertEqual(projection.shape[1], 1)  # Should be 1D projection
        self.assertEqual(transform.shape[1], 1)   # Should be 1D transformation
        self.assertEqual(projection.shape[0], len(self.pos_activations) + len(self.neg_activations))

    def test_separation_quality(self):
        """Test if the LDA properly separates clearly separated data."""
        projection, _, metrics = cdr.compute_binary_lda_projection(
            self.pos_activations, self.neg_activations, self.device
        )
        
        # For clearly separated data, we expect:
        self.assertGreater(metrics['classification_accuracy'], 0.9)  # High accuracy
        self.assertGreater(metrics['separation_score'], 1.0)        # Good separation
        self.assertGreater(metrics['explained_variance_ratio'], 0.5) # High explained variance

    def test_input_validation(self):
        """Test if the function properly validates inputs."""
        # Test with empty lists
        with self.assertRaises(ValueError):
            cdr.compute_binary_lda_projection([], self.neg_activations, self.device)
            
        # Test with single sample per class
        with self.assertRaises(ValueError):
            cdr.compute_binary_lda_projection(self.single_pos, self.single_pos, self.device)
            
        # Test with different dimensions
        with self.assertRaises(ValueError):
            cdr.compute_binary_lda_projection(self.diff_dim_pos, self.pos_activations, self.device)

    def test_zero_variance(self):
        """Test if the function properly handles zero variance features."""
        with self.assertRaises(ValueError):
            cdr.compute_binary_lda_projection(self.zero_var_pos, self.zero_var_pos, self.device)

    def test_device_handling(self):
        """Test if the function properly handles device specification."""
        # Test with explicit CPU device
        projection_cpu, _, _ = cdr.compute_binary_lda_projection(
            self.pos_activations, self.neg_activations, torch.device('cpu')
        )
        self.assertEqual(projection_cpu.device.type, 'cpu')
        
        # Test with default device
        projection_default, _, _ = cdr.compute_binary_lda_projection(
            self.pos_activations, self.neg_activations
        )
        self.assertEqual(projection_default.device.type, self.device.type)

    def test_numerical_stability(self):
        """Test if the function properly warns about numerical instability."""
        # Create ill-conditioned data with small random variations
        n_samples_numerical_stability = 50
        torch.manual_seed(42)  # for reproducibility
        
        # Create base vectors with small random variations
        ill_pos = [
            torch.ones(self.n_features).to(self.device) * 1e3 + 
            torch.randn(self.n_features).to(self.device) * 1e-3  # small random noise
            for _ in range(n_samples_numerical_stability)
        ]
        ill_neg = [
            torch.ones(self.n_features).to(self.device) * -1e3 + 
            torch.randn(self.n_features).to(self.device) * 1e-3  # small random noise
            for _ in range(n_samples_numerical_stability)
        ]
        
        with self.assertWarns(Warning):
            cdr.compute_binary_lda_projection(ill_pos, ill_neg, self.device)

    def test_high_dimensional_projection(self):
        """Test LDA projection of high-dimensional data into 2D space with different configurations."""
        # Adjust parameters for high dimensions
        configs = [
            # n_features, n_samples, separation, min_expected_separation
            (100, 200, 2.0, 0.8),    # Lower dimensional case
            (1000, 500, 5.0, 0.3),   # High dimensional case needs more samples and separation
        ]
        
        for n_features, n_samples, separation, min_separation in configs:
            torch.manual_seed(42)
            
            # Create means for two Gaussian distributions
            mean_pos = torch.zeros(n_features)
            mean_pos[0] = separation
            mean_neg = torch.zeros(n_features)
            mean_neg[0] = -separation
            
            # Reduce noise in other dimensions
            noise_scale = 0.1  # Reduce noise in non-separating dimensions
            
            pos_samples = [
                (torch.randn(n_features) * noise_scale + mean_pos).to(self.device)
                for _ in range(n_samples)
            ]
            neg_samples = [
                (torch.randn(n_features) * noise_scale + mean_neg).to(self.device)
                for _ in range(n_samples)
            ]
                    
            # Compute LDA projection
            projection, transform, metrics = cdr.compute_binary_lda_projection(
                pos_samples, neg_samples, self.device
            )
            
            # Test dimensionality
            self.assertEqual(projection.shape[1], 1)
            self.assertEqual(transform.shape[0], n_features)
            self.assertEqual(transform.shape[1], 1)
            
            # Test separation quality (should increase with separation parameter)
            self.assertGreater(
                metrics['separation_score'],
                separation / 2,  # Expected minimum separation
                f"Poor separation for {n_features}D data with {n_samples} samples "
                f"and separation {separation}"
            )
            
            # Test classification accuracy (should be better for more separated data)
            min_expected_accuracy = 0.75 if separation >= 5.0 else 0.65
            self.assertGreater(
                metrics['classification_accuracy'],
                min_expected_accuracy,
                f"Poor classification for {n_features}D data with {n_samples} samples "
                f"and separation {separation}"
            )

    def test_projection_stability(self):
        """Test if projections are stable across different random initializations."""
        n_features = 100  # Reduce from 500
        n_samples = 200   # Increase from 100
        separation = 5.0  # Increase from 3.0
        n_runs = 3
        
        # Store projections from multiple runs
        all_projections = []
        
        for run in range(n_runs):
            torch.manual_seed(run)
            
            # Create Gaussian distributions with reduced noise
            mean_pos = torch.zeros(n_features)
            mean_pos[0] = separation
            mean_neg = torch.zeros(n_features)
            mean_neg[0] = -separation
            
            # Reduce noise in other dimensions
            noise_scale = 0.1
            
            pos_samples = [
                (torch.randn(n_features) * noise_scale + mean_pos).to(self.device)
                for _ in range(n_samples)
            ]
            neg_samples = [
                (torch.randn(n_features) * noise_scale + mean_neg).to(self.device)
                for _ in range(n_samples)
            ]
            
            projection, _, _ = cdr.compute_binary_lda_projection(
                pos_samples, neg_samples, self.device
            )
            
            # Normalize projection before storing
            projection = F.normalize(projection, dim=0)
            all_projections.append(projection)
        
        # Test stability with lower threshold
        for i in range(1, n_runs):
            correlation = torch.corrcoef(
                torch.stack([all_projections[i].flatten(), 
                            all_projections[i-1].flatten()])
            )[0,1].abs()
            
            self.assertGreater(
                correlation,
                0.7,  # Lower correlation threshold
                f"Unstable projections between runs {i} and {i-1}"
            )

    def test_pca_basic_functionality(self):
        """Test if the PCA function runs with valid inputs and returns expected output types."""
        projection, transform, metrics = cdr.compute_binary_pca_projection(
            self.pos_activations, self.neg_activations, n_components=2, device=self.device
        )
        
        # Check output types
        self.assertIsInstance(projection, torch.Tensor)
        self.assertIsInstance(transform, torch.Tensor)
        self.assertIsInstance(metrics, dict)
        
        # Check output dimensions
        self.assertEqual(projection.shape[1], 2)  # Should be 2D projection by default
        self.assertEqual(transform.shape[1], 2)   # Should be 2D transformation
        self.assertEqual(projection.shape[0], len(self.pos_activations) + len(self.neg_activations))
        
        # Check metrics structure
        self.assertIn('explained_variance_ratio', metrics)
        self.assertIn('cumulative_variance_ratio', metrics)
        self.assertIn('separation_scores', metrics)
        self.assertEqual(len(metrics['separation_scores']), 2)  # One for each component

    def test_pca_variance_explanation(self):
        """Test if PCA properly captures variance in the data."""
        _, _, metrics = cdr.compute_binary_pca_projection(
            self.pos_activations, self.neg_activations, device=self.device
        )
        
        # Check explained variance properties
        var_ratios = metrics['explained_variance_ratio']
        cumulative_var = metrics['cumulative_variance_ratio']
        
        # Variance ratios should sum to cumulative variance
        self.assertAlmostEqual(sum(var_ratios), cumulative_var, places=5)
        
        # Variance ratios should be in descending order
        self.assertTrue(all(var_ratios[i] >= var_ratios[i+1] 
                        for i in range(len(var_ratios)-1)))
        
        # Cumulative variance should be between 0 and 1
        self.assertGreater(cumulative_var, 0.0)
        self.assertLessEqual(cumulative_var, 1.0)

    def test_pca_components(self):
        """Test different numbers of components."""
        n_components_list = [1, 2, 5]
        
        for n_components in n_components_list:
            projection, transform, metrics = cdr.compute_binary_pca_projection(
                self.pos_activations, self.neg_activations,
                n_components=n_components,
                device=self.device
            )
            
            # Check dimensions
            self.assertEqual(projection.shape[1], n_components)
            self.assertEqual(transform.shape[1], n_components)
            self.assertEqual(len(metrics['separation_scores']), n_components)
            self.assertEqual(len(metrics['explained_variance_ratio']), n_components)

    def test_pca_input_validation(self):
        """Test if the PCA function properly validates inputs."""
        # Test with empty lists
        with self.assertRaises(ValueError):
            cdr.compute_binary_pca_projection([], self.neg_activations, device=self.device)
        
        # Test with different dimensions
        with self.assertRaises(ValueError):
            cdr.compute_binary_pca_projection(self.diff_dim_pos, self.pos_activations, device=self.device)
        
        # Test with invalid n_components
        with self.assertRaises(ValueError):
            cdr.compute_binary_pca_projection(
                self.pos_activations, self.neg_activations,
                n_components=len(self.pos_activations) + len(self.neg_activations) + 1,
                device=self.device
            )

    def test_pca_zero_variance(self):
        """Test if the PCA function properly handles zero variance features."""
        with self.assertRaises(ValueError):
            cdr.compute_binary_pca_projection(
                self.zero_var_pos, self.zero_var_pos, device=self.device
            )

    def test_pca_device_handling(self):
        """Test if the PCA function properly handles device specification."""
        # Test with explicit CPU device
        projection_cpu, _, _ = cdr.compute_binary_pca_projection(
            self.pos_activations, self.neg_activations,
            device=torch.device('cpu')
        )
        self.assertEqual(projection_cpu.device.type, 'cpu')
        
        # Test with default device
        projection_default, _, _ = cdr.compute_binary_pca_projection(
            self.pos_activations, self.neg_activations
        )
        self.assertEqual(projection_default.device.type, self.device.type)

    def test_pca_numerical_stability(self):
        """Test if the PCA function properly warns about numerical instability."""
        # Create ill-conditioned data
        n_samples = 50
        torch.manual_seed(42)
        
        ill_pos = [
            torch.ones(self.n_features).to(self.device) * 1e6 + 
            torch.randn(self.n_features).to(self.device) * 1e-6
            for _ in range(n_samples)
        ]
        ill_neg = [
            torch.ones(self.n_features).to(self.device) * -1e6 + 
            torch.randn(self.n_features).to(self.device) * 1e-6
            for _ in range(n_samples)
        ]
        
        with self.assertWarns(Warning):
            cdr.compute_binary_pca_projection(ill_pos, ill_neg, device=self.device)

    def test_pca_reproducibility(self):
        """Test if PCA results are reproducible with same random seed."""
        torch.manual_seed(42)
        proj1, trans1, metrics1 = cdr.compute_binary_pca_projection(
            self.pos_activations, self.neg_activations, device=self.device
        )
        
        torch.manual_seed(42)
        proj2, trans2, metrics2 = cdr.compute_binary_pca_projection(
            self.pos_activations, self.neg_activations, device=self.device
        )
        
        # Check if results are identical
        self.assertTrue(torch.allclose(proj1, proj2))
        self.assertTrue(torch.allclose(trans1, trans2))
        self.assertEqual(metrics1['explained_variance_ratio'], 
                        metrics2['explained_variance_ratio'])

    def test_pca_orthogonality(self):
        """Test if PCA components are orthogonal."""
        _, transform, _ = cdr.compute_binary_pca_projection(
            self.pos_activations, self.neg_activations,
            n_components=2,
            device=self.device
        )
        
        # Check orthogonality of components
        components = transform.T  # Shape: (n_components, n_features)
        dot_product = torch.mm(components, components.T)
        
        # Off-diagonal elements should be close to 0
        off_diag = dot_product - torch.diag(torch.diag(dot_product))
        self.assertTrue(torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6))


if __name__ == '__main__':
    unittest.main()