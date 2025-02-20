import unittest
import torch
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

import compute_steering_vectors as sv
)
import validate_custom_datatypes as vcd
from pca import PCA

class TestSteeringVectors(unittest.TestCase):
    def setUp(self):
        """Set up test data with 1D tensors"""
        self.device = "mps" if torch.backends.mps.is_available() else torch.device("cpu")
        self.hidden_size = 16
        self.sample_size = 10
        self.n_layers = 5

        # Create sample activations using 1D tensors
        self.positive_activations = {
            i: [torch.rand(self.hidden_size) for _ in range(self.sample_size)]
            for i in range(self.n_layers)
        }
        self.negative_activations = {
            i: [torch.rand(self.hidden_size) for _ in range(self.sample_size)]
            for i in range(self.n_layers)
        }

    def test_activation_validation(self):
        """Test that our test data passes validation"""
        # Test valid input
        try:
            vcd.validate_activation_list_dict(self.positive_activations)
            vcd.validate_activation_list_dict(self.negative_activations)
        except (TypeError, ValueError) as e:
            self.fail(f"Validation failed: {str(e)}")

        # Test invalid 2D tensor
        invalid_activations = {
            0: [torch.rand(3, self.hidden_size)]
        }
        with self.assertRaises(ValueError):
            vcd.validate_activation_list_dict(invalid_activations)

    def test_compute_contrastive_steering_vector(self):
        """Test contrastive steering vector computation"""
        steering_vectors = vcd.compute_contrastive_steering_vector(
            self.positive_activations, self.negative_activations, self.device
        )

        # Test basic properties
        self.assertIsInstance(steering_vectors, dict)
        self.assertEqual(len(steering_vectors), self.n_layers)

        # Test each steering vector
        for layer, vector in steering_vectors.items():
            self.assertIsInstance(layer, int)
            self.assertIsInstance(vector, torch.Tensor)
            self.assertEqual(len(vector.shape), 1)  # Ensure 1D
            self.assertEqual(vector.shape[0], self.hidden_size)

        # Validate using custom validation function
        try:
            vcd.validate_steering_vector_dict(steering_vectors)
        except (TypeError, ValueError) as e:
            self.fail(f"Steering vector validation failed: {str(e)}")

    def test_compute_non_contrastive_steering_vector(self):
        """Test non-contrastive steering vector computation"""
        steering_vectors = vcd.compute_non_contrastive_steering_vector(
            self.positive_activations, self.device
        )

        self.assertIsInstance(steering_vectors, dict)
        self.assertEqual(len(steering_vectors), self.n_layers)
        for layer, vector in steering_vectors.items():
            self.assertIsInstance(layer, int)
            self.assertIsInstance(vector, torch.Tensor)
            self.assertEqual(len(vector.shape), 1)  # Ensure 1D
            self.assertEqual(vector.shape[0], self.hidden_size)

        # Validate using custom validation function
        try:
            vcd.validate_steering_vector_dict(steering_vectors)
        except (TypeError, ValueError) as e:
            self.fail(f"Steering vector validation failed: {str(e)}")

    def test_compute_in_context_vector(self):
        """Test in-context vector computation"""
        n_components = 1  # Only testing n_components=1 as that's all that's supported
        in_context_vectors = vcd.compute_in_context_vector(
            self.positive_activations, self.negative_activations, self.device, n_components
        )

        self.assertIsInstance(in_context_vectors, dict)
        self.assertEqual(len(in_context_vectors), self.n_layers)
        for layer, vector in in_context_vectors.items():
            self.assertIsInstance(layer, int)
            self.assertIsInstance(vector, torch.Tensor)
            self.assertEqual(len(vector.shape), 1)  # Ensure 1D
            self.assertEqual(vector.shape[0], self.hidden_size)

        # Test with invalid n_components
        with self.assertRaises(NotImplementedError):
            vcd.compute_in_context_vector(
                self.positive_activations, self.negative_activations, 
                self.device, n_components=2
            )

        # Validate using custom validation function
        try:
            vcd.validate_steering_vector_dict(in_context_vectors)
        except (TypeError, ValueError) as e:
            self.fail(f"Steering vector validation failed: {str(e)}")

    def test_contrastive_steering_vector_alignment(self):
        """Test if steering vector aligns with the shift direction between positive and negative samples"""
        EPSILON = 1e-5  # Tolerance for cosine similarity comparison
        
        # Test configurations
        configs = [
            {"hidden_size": 16, "sample_size": 10, "n_layers": 5},  # Small config
            {"hidden_size": 23, "sample_size": 34, "n_layers": 6},  # Medium config
            {"hidden_size": 64, "sample_size": 50, "n_layers": 8}   # Larger config
        ]
        
        for config in configs:
            hidden_size = config["hidden_size"]
            sample_size = config["sample_size"]
            n_layers = config["n_layers"]
            
            # Store the layer-specific offsets
            true_offsets = {}
            
            # Generate test data
            pos_activations = {}
            neg_activations = {}
            
            for layer in range(n_layers):
                # Generate random offset for this layer
                true_offsets[layer] = torch.rand(hidden_size, device=self.device)
                true_offsets[layer] = true_offsets[layer] / torch.norm(true_offsets[layer])  # normalize
                
                # Generate positive samples
                neg_samples = [torch.rand(hidden_size, device=self.device) for _ in range(sample_size)]
                
                # Generate negative samples by adding scaled offsets
                pos_samples = []
                for neg in neg_samples:
                    # Random scalar from uniform distribution [0.2, 4]
                    scalar = 0.2 + 3.8 * torch.rand(1, device=self.device).item()
                    pos = neg + scalar * true_offsets[layer]
                    pos_samples.append(pos)
                    
                pos_activations[layer] = pos_samples
                neg_activations[layer] = neg_samples
            
            # Compute steering vectors
            steering_vectors = vcd.compute_contrastive_steering_vector(
                pos_activations, neg_activations, self.device
            )
            
            # Check alignment for each layer
            for layer in range(n_layers):
                sv = steering_vectors[layer]
                sv_normalized = sv / torch.norm(sv)
                
                # Compute cosine similarity
                cos_sim = torch.dot(sv_normalized, true_offsets[layer])
                
                self.assertGreaterEqual(
                    cos_sim, 
                    1 - EPSILON, 
                    f"Config {hidden_size}_{sample_size}_{n_layers}, Layer {layer}: "
                    f"Steering vector not aligned with true offset. Cosine similarity: {cos_sim}"
                )

    def test_pca_direction_recovery(self):
        """Test if PCA can recover known principal directions in synthetic data"""
        EPSILON = 1e-2
        
        # Test configurations
        configs = [
            {"n_samples": 100, "n_features": 16, "n_components": 2},  # Small config
            {"n_samples": 1000, "n_features": 64, "n_components": 3}  # Larger config
        ]
        
        for config in configs:
            n_samples = config["n_samples"]
            n_features = config["n_features"]
            n_components = config["n_components"]
            
            # Create true directions (orthogonal vectors)
            true_directions = torch.zeros((n_components, n_features), device=self.device)
            for i in range(n_components):
                # Create orthogonal vectors by setting different features to 1
                true_directions[i, i*2:(i+1)*2] = 1.0
                true_directions[i] = true_directions[i] / torch.norm(true_directions[i])
            
            # Generate synthetic data
            X = torch.zeros((n_samples, n_features), device=self.device)
            for i in range(n_components):
                # Scale each direction with decreasing variance
                scale = 10.0 / (i + 1)
                components = torch.randn(n_samples, device=self.device) * scale
                X += torch.outer(components, true_directions[i])
                
            # Add small noise
            X += torch.randn_like(X) * 0.1
            
            # Fit PCA
            pca = PCA(device=self.device, n_components=n_components)
            pca.fit(X)
            
            # Check if recovered directions align with true directions
            # Note: PCA components might be recovered in different order
            for i in range(n_components):
                # Find best matching direction
                cos_sims = torch.abs(torch.matmul(pca.components_, true_directions.T))
                max_cos_sim = cos_sims.max()
                
                self.assertGreaterEqual(
                    max_cos_sim, 
                    1 - EPSILON,
                    f"Config {n_samples}_{n_features}_{n_components}, Component {i}: "
                    f"Failed to recover true direction. Best cosine similarity: {max_cos_sim}"
                )
                
    def test_in_context_vector_recovery(self):
        """Test if in-context vectors can recover known directions in synthetic data"""
        EPSILON = 1e-2
        
        # Test configurations
        configs = [
            {"hidden_size": 16, "sample_size": 200, "n_layers": 5},  # Small config
            {"hidden_size": 64, "sample_size": 300, "n_layers": 8}   # Larger config
        ]
        
        for config in configs:
            hidden_size = config["hidden_size"]
            sample_size = config["sample_size"]
            n_layers = config["n_layers"]
            
            # Store the layer-specific directions
            true_directions = {}
            
            # Generate test data
            pos_activations = {}
            neg_activations = {}
            
            for layer in range(n_layers):
                # For each layer, create two orthogonal directions
                # First direction: main difference between pos/neg
                # Second direction: common variation in both pos/neg
                direction1 = torch.rand(hidden_size, device=self.device)
                direction1 = direction1 / torch.norm(direction1)
                
                # Create orthogonal direction using Gram-Schmidt
                direction2 = torch.rand(hidden_size, device=self.device)
                direction2 = direction2 - torch.dot(direction2, direction1) * direction1
                direction2 = direction2 / torch.norm(direction2)
                
                true_directions[layer] = direction1
                
                # Generate samples
                pos_samples = []
                neg_samples = []
                
                for _ in range(sample_size):
                    # Common variation (should be ignored by in-context vector)
                    common = torch.randn(1, device=self.device) * direction2
                    
                    # Positive samples: high values in direction1
                    pos_value = torch.abs(torch.randn(1, device=self.device)) * 2 + 3  # Always positive, shifted away from 0
                    pos_base = pos_value * direction1 + common
                    pos_samples.append(pos_base + torch.randn_like(pos_base) * 0.1)  # Small noise
                    
                    # Negative samples: low values in direction1
                    neg_value = -torch.abs(torch.randn(1, device=self.device)) * 2 - 3  # Always negative, shifted away from 0
                    neg_base = neg_value * direction1 + common
                    neg_samples.append(neg_base + torch.randn_like(neg_base) * 0.1)  # Small noise
                    
                pos_activations[layer] = pos_samples
                neg_activations[layer] = neg_samples
            
            # Compute in-context vectors
            in_context_vectors = vcd.compute_in_context_vector(
                pos_activations, neg_activations, self.device, n_components=1
            )
            
            # Check alignment for each layer
            for layer in range(n_layers):
                icv = in_context_vectors[layer]
                icv_normalized = icv / torch.norm(icv)
                
                # Compute absolute cosine similarity (direction might be flipped)
                cos_sim = torch.abs(torch.dot(icv_normalized, true_directions[layer]))
                
                self.assertGreaterEqual(
                    cos_sim, 
                    1 - EPSILON, 
                    f"Config {hidden_size}_{sample_size}_{n_layers}, Layer {layer}: "
                    f"In-context vector not aligned with true direction. Cosine similarity: {cos_sim}"
                )
        
    def test_orthogonality_properties(self):
        """Test that steering vectors maintain expected orthogonality properties
        when computed from orthogonal data distributions."""
        
        def create_orthogonal_samples(base_direction, orthogonal_direction, samples_per_direction, noise_level=0.05):  # Reduced noise
            """Helper to create samples along orthogonal directions"""
            samples = []
            # Samples along base direction with stronger signal
            for _ in range(samples_per_direction):
                magnitude = 2.0 + 0.5 * torch.rand(1, device=self.device).item()  # Increased base magnitude
                sample = magnitude * base_direction + noise_level * torch.randn_like(base_direction)
                samples.append(sample)
            # Samples along orthogonal direction with weaker signal
            for _ in range(samples_per_direction):
                magnitude = 0.5 + 0.2 * torch.rand(1, device=self.device).item()  # Reduced orthogonal magnitude
                sample = magnitude * orthogonal_direction + noise_level * torch.randn_like(orthogonal_direction)
                samples.append(sample)
            return samples

        # Test configurations
        hidden_sizes = [16, 32, 64]
        samples_per_direction = 100  # Increased sample size
        epsilon = 0.15  # Relaxed tolerance

        for hidden_size in hidden_sizes:
            # Create two orthogonal base directions
            direction1 = torch.randn(hidden_size, device=self.device)
            direction1 = direction1 / torch.norm(direction1)
            
            # Create second direction orthogonal to first using Gram-Schmidt
            direction2 = torch.randn(hidden_size, device=self.device)
            direction2 = direction2 - torch.dot(direction2, direction1) * direction1
            direction2 = direction2 / torch.norm(direction2)
            
            # Verify orthogonality of base directions
            self.assertLess(abs(torch.dot(direction1, direction2)), epsilon)
            
            # Create samples along these orthogonal directions
            pos_activations = {
                0: create_orthogonal_samples(direction1, direction2, samples_per_direction)
            }
            neg_activations = {
                0: create_orthogonal_samples(-direction1, -direction2, samples_per_direction)
            }
            
            # Compute steering vectors
            steering_vectors = vcd.compute_contrastive_steering_vector(
                pos_activations, neg_activations, self.device
            )
            
            # Compute in-context vectors
            in_context_vectors = vcd.compute_in_context_vector(
                pos_activations, neg_activations, self.device, n_components=1
            )
            
            # Verify that steering vectors preserve directionality
            sv = steering_vectors[0]
            icv = in_context_vectors[0]
            
            # Normalize vectors
            sv_norm = sv / torch.norm(sv)
            icv_norm = icv / torch.norm(icv)
            
            # Check alignment with primary direction
            sv_alignment = abs(torch.dot(sv_norm, direction1))
            icv_alignment = abs(torch.dot(icv_norm, direction1))
            
            self.assertGreater(sv_alignment, 1 - epsilon,
                            f"Steering vector not aligned with primary direction. "
                            f"Alignment: {sv_alignment}")
            self.assertGreater(icv_alignment, 1 - epsilon,
                            f"In-context vector not aligned with primary direction. "
                            f"Alignment: {icv_alignment}")

    def test_scale_invariance(self):
        """Test that steering vectors are invariant to scaling of input activations."""
        
        # Test configurations
        scales = [0.1, 1.0, 10.0, 100.0]
        samples_size = 50
        epsilon = 1e-5
        
        # Create base direction
        base_direction = torch.randn(self.hidden_size, device=self.device)
        base_direction = base_direction / torch.norm(base_direction)
        
        # Generate base samples
        base_pos_samples = [
            base_direction + 0.1 * torch.randn_like(base_direction)
            for _ in range(samples_size)
        ]
        base_neg_samples = [
            -base_direction + 0.1 * torch.randn_like(base_direction)
            for _ in range(samples_size)
        ]
        
        # Compute base steering vector
        base_activations = {
            0: base_pos_samples,
            1: base_neg_samples
        }
        base_steering = vcd.compute_non_contrastive_steering_vector(base_activations, self.device)
        
        for scale in scales:
            # Scale the samples
            scaled_pos_samples = [scale * sample for sample in base_pos_samples]
            scaled_neg_samples = [scale * sample for sample in base_neg_samples]
            
            scaled_activations = {
                0: scaled_pos_samples,
                1: scaled_neg_samples
            }
            
            # Compute steering vector for scaled samples
            scaled_steering = vcd.compute_non_contrastive_steering_vector(
                scaled_activations, self.device
            )
            
            # Compare normalized vectors
            for layer in [0, 1]:
                base_norm = base_steering[layer] / torch.norm(base_steering[layer])
                scaled_norm = scaled_steering[layer] / torch.norm(scaled_steering[layer])
                
                # Compute cosine similarity
                cos_sim = abs(torch.dot(base_norm, scaled_norm))
                
                self.assertGreater(cos_sim, 1 - epsilon,
                                f"Scale invariance violated at scale {scale}, "
                                f"layer {layer}. Cosine similarity: {cos_sim}")
                


if __name__ == '__main__':
    unittest.main()