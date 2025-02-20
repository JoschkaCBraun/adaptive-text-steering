import os
import sys
import math
import unittest
import torch
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

import compute_properties_of_activations as cpa

from validate_custom_datatypes import validate_activation_list

class TestActivationComputations(unittest.TestCase):
    def setUp(self):
        """Set up test cases with sample activation data for both list and dictionary tests."""
        # Use MPS if available, otherwise CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create valid test tensors
        self.valid_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        self.valid_tensor_2 = torch.tensor([4.0, 5.0, 6.0], device=self.device)
        self.valid_tensor_3 = torch.tensor([7.0, 8.0, 9.0], device=self.device)
        
        # List-based test data
        self.valid_activations = [self.valid_tensor_1, self.valid_tensor_2, self.valid_tensor_3]
        self.steering_vector = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        
        # Dictionary-based test data
        self.valid_activations_dict = {
            0: [self.valid_tensor_1, self.valid_tensor_2, self.valid_tensor_3],
            1: [self.valid_tensor_2, self.valid_tensor_3, self.valid_tensor_1],
        }
        self.valid_steering_dict = {
            0: torch.tensor([1.0, 1.0, 1.0], device=self.device),
            1: torch.tensor([2.0, 2.0, 2.0], device=self.device)
        }
        
        # Invalid test cases
        self.wrong_shape_tensor = torch.tensor([1.0, 2.0], device=self.device)
        self.wrong_dim_tensor = torch.tensor([[1.0, 2.0, 3.0]], device=self.device)
        self.non_tensor = [1.0, 2.0, 3.0]
        self.empty_list = []
        self.empty_dict = {}
        self.empty_layer_dict = {
            0: [],
            1: [self.valid_tensor_1, self.valid_tensor_2]
        }
        self.mismatched_layers_dict = {
            0: [self.valid_tensor_1, self.valid_tensor_2],
            2: [self.valid_tensor_2, self.valid_tensor_3]  # Layer 1 missing
        }

    # List-based tests
    def test_pairwise_cosine_similarity(self):
        """Test computation of pairwise cosine similarities with lists."""
        # Test with valid inputs
        similarities = cpa.compute_pairwise_cosine_similarity(self.valid_activations)
        self.assertEqual(len(similarities), 3)  # Should have 3 unique pairs for 3 vectors
        
        # Verify values are between -1 and 1
        for sim in similarities:
            self.assertTrue(-1.0 <= sim <= 1.0)
        
        # Test with single vector
        single_vector = [self.valid_tensor_1]
        similarities = cpa.compute_pairwise_cosine_similarity(single_vector)
        self.assertEqual(len(similarities), 0)  # No pairs with single vector
        
        # Test error cases
        with self.assertRaises(ValueError):
            cpa.compute_pairwise_cosine_similarity(self.empty_list)
        
        with self.assertRaises(TypeError):
            cpa.compute_pairwise_cosine_similarity([self.non_tensor])
        
        with self.assertRaises(ValueError):
            cpa.compute_pairwise_cosine_similarity([self.wrong_shape_tensor, self.valid_tensor_1])

    def test_cosine_similarity_many_against_one(self):
        """Test computation of cosine similarities against steering vector with lists."""
        # Test with valid inputs
        similarities = cpa.compute_many_against_one_cosine_similarity(
            self.valid_activations, self.steering_vector
        )
        self.assertEqual(len(similarities), len(self.valid_activations))
        
        # Verify values are between -1 and 1
        for sim in similarities:
            self.assertTrue(-1.0 <= sim <= 1.0)
        
        # Test error cases
        with self.assertRaises(ValueError):
            cpa.compute_many_against_one_cosine_similarity(
                self.empty_list, self.steering_vector
            )
        
        wrong_size_steering = torch.tensor([1.0, 1.0], device=self.device)
        with self.assertRaises(ValueError):
            cpa.compute_many_against_one_cosine_similarity(
                self.valid_activations, wrong_size_steering
            )

    def test_l2_norms(self):
        """Test computation of L2 norms with lists."""
        # Test with valid inputs
        norms = cpa.compute_l2_norms(self.valid_activations)
        self.assertEqual(len(norms), len(self.valid_activations))
        
        # Verify first norm manually
        expected_norm = math.sqrt(sum(x*x for x in [1.0, 2.0, 3.0]))
        self.assertAlmostEqual(norms[0], expected_norm, places=5)
        
        # Test error cases
        with self.assertRaises(ValueError):
            cpa.compute_l2_norms(self.empty_list)
        
        with self.assertRaises(TypeError):
            cpa.compute_l2_norms([self.non_tensor])
        
        with self.assertRaises(ValueError):
            cpa.compute_l2_norms([self.wrong_dim_tensor])

    # Dictionary-based tests
    def test_pairwise_cosine_similarity_dict(self):
        """Test computation of pairwise cosine similarities with dictionaries."""
        # Test with valid inputs
        similarities = cpa.compute_pairwise_cosine_similarity_dict(self.valid_activations_dict)
        
        # Check dictionary structure
        self.assertEqual(set(similarities.keys()), set(self.valid_activations_dict.keys()))
        
        # Check each layer's similarities
        for layer in similarities:
            self.assertEqual(len(similarities[layer]), 3)  # Should have 3 unique pairs
            for sim in similarities[layer]:
                self.assertTrue(-1.0 <= sim <= 1.0)
        
        # Test error cases
        with self.assertRaises(ValueError):
            cpa.compute_pairwise_cosine_similarity_dict(self.empty_dict)
        
        with self.assertRaises(ValueError):
            cpa.compute_pairwise_cosine_similarity_dict(self.empty_layer_dict)

    def test_many_against_one_cosine_similarity_dict(self):
        """Test computation of cosine similarities against steering vectors with dictionaries."""
        # Test with valid inputs
        similarities = cpa.compute_many_against_one_cosine_similarity_dict(
            self.valid_activations_dict, self.valid_steering_dict
        )
        
        # Check dictionary structure
        self.assertEqual(set(similarities.keys()), set(self.valid_activations_dict.keys()))
        
        # Check each layer's similarities
        for layer in similarities:
            self.assertEqual(len(similarities[layer]), 
                           len(self.valid_activations_dict[layer]))
            for sim in similarities[layer]:
                self.assertTrue(-1.0 <= sim <= 1.0)
        
        # Test with mismatched layers
        mismatched_steering_dict = {0: self.valid_steering_dict[0]}
        with self.assertRaises(ValueError):
            cpa.compute_many_against_one_cosine_similarity_dict(
                self.valid_activations_dict, mismatched_steering_dict
            )

    def test_l2_norms_dict(self):
        """Test computation of L2 norms with dictionaries."""
        # Test with valid inputs
        norms = cpa.compute_l2_norms_dict(self.valid_activations_dict)
        
        # Check dictionary structure
        self.assertEqual(set(norms.keys()), set(self.valid_activations_dict.keys()))
        
        # Check each layer's norms
        for layer in norms:
            self.assertEqual(len(norms[layer]), len(self.valid_activations_dict[layer]))
            if layer == 0:
                expected_norm = math.sqrt(sum(x*x for x in [1.0, 2.0, 3.0]))
                self.assertAlmostEqual(norms[layer][0], expected_norm, places=5)
        
        # Test error cases
        with self.assertRaises(ValueError):
            cpa.compute_l2_norms_dict(self.empty_dict)

    # Device handling tests
    def test_device_handling(self):
        """Test handling of different devices for both list and dictionary inputs."""
        cpu_device = torch.device("cpu")
        
        # Test list-based functions
        list_similarities = cpa.compute_pairwise_cosine_similarity(
            self.valid_activations, device=cpu_device
        )
        self.assertEqual(len(list_similarities), 3)
        
        # Test dictionary-based functions
        dict_similarities = cpa.compute_pairwise_cosine_similarity_dict(
            self.valid_activations_dict, device=cpu_device
        )
        self.assertEqual(len(dict_similarities), len(self.valid_activations_dict))
        
        # Test with mixed devices
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
        
        # Test lists with mixed devices
        mixed_devices_list = [self.valid_tensor_1, cpu_tensor]
        with self.assertRaises(ValueError):
            cpa.compute_pairwise_cosine_similarity(mixed_devices_list)
        
        # Test dictionaries with mixed devices
        mixed_devices_dict = {
            0: [self.valid_tensor_1, cpu_tensor],
            1: [self.valid_tensor_2, self.valid_tensor_3]
        }
        with self.assertRaises(ValueError):
            cpa.compute_pairwise_cosine_similarity_dict(mixed_devices_dict)

if __name__ == '__main__':
    unittest.main()