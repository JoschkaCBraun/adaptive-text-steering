import torch
import torch.nn.functional as F
import itertools
import time
from typing import List, Optional

def compute_pairwise_cosine_similarity_original(
        activations: List[torch.Tensor], device: Optional[torch.device] = "cpu") -> List[float]:
    """Original implementation using triu_indices."""
    if device is None:
        device = activations[0].device
    
    vectors_stacked = torch.stack([activation.to(device) for activation in activations])
    vectors_normalized = F.normalize(vectors_stacked, p=2, dim=1)
    similarity_matrix = torch.mm(vectors_normalized, vectors_normalized.T)
    
    # Get upper triangular part using triu_indices
    indices = torch.triu_indices(len(activations), len(activations), offset=1, device=device)
    similarities = similarity_matrix[indices.unbind()].tolist()
    return similarities

def compute_pairwise_cosine_similarity_loops(
        activations: List[torch.Tensor], device: Optional[torch.device] = None) -> List[float]:
    """Implementation using nested loops - simple but slower for large inputs."""
    if device is None:
        device = activations[0].device
        
    vectors = [activation.to(device) for activation in activations]
    similarities = []
    
    for i in range(len(vectors)):
        v1_normalized = F.normalize(vectors[i].unsqueeze(0), p=2, dim=1)
        for j in range(i + 1, len(vectors)):
            v2_normalized = F.normalize(vectors[j].unsqueeze(0), p=2, dim=1)
            sim = torch.mm(v1_normalized, v2_normalized.T).item()
            similarities.append(sim)
            
    return similarities

def compute_pairwise_cosine_similarity_mask(
        activations: List[torch.Tensor], device: Optional[torch.device] = 'cpu') -> List[float]:
    """Implementation using a boolean mask - memory efficient."""
    if device is None:
        device = activations[0].device
        
    vectors_stacked = torch.stack([activation.to(device) for activation in activations])
    vectors_normalized = F.normalize(vectors_stacked, p=2, dim=1)
    similarity_matrix = torch.mm(vectors_normalized, vectors_normalized.T)
    
    # Create boolean mask for upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    similarities = similarity_matrix[mask].tolist()
    return similarities

def compute_pairwise_cosine_similarity_combinations(
        activations: List[torch.Tensor], device: Optional[torch.device] = None) -> List[float]:
    """Implementation using itertools.combinations - good balance of simplicity and speed."""
    if device is None:
        device = activations[0].device
        
    vectors = [activation.to(device) for activation in activations]
    similarities = []
    
    for v1, v2 in itertools.combinations(vectors, 2):
        v1_normalized = F.normalize(v1.unsqueeze(0), p=2, dim=1)
        v2_normalized = F.normalize(v2.unsqueeze(0), p=2, dim=1)
        sim = torch.mm(v1_normalized, v2_normalized.T).item()
        similarities.append(sim)
        
    return similarities

def compute_pairwise_cosine_similarity_einsum(
        activations: List[torch.Tensor], device: Optional[torch.device] = 'cpu') -> List[float]:
    """Implementation using einsum - potentially faster for large tensors."""
    if device is None:
        device = activations[0].device
        
    vectors_stacked = torch.stack([activation.to(device) for activation in activations])
    vectors_normalized = F.normalize(vectors_stacked, p=2, dim=1)
    
    # Use einsum for matrix multiplication
    similarity_matrix = torch.einsum('ik,jk->ij', vectors_normalized, vectors_normalized)
    
    # Create mask for upper triangle
    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    similarities = similarity_matrix[mask].tolist()
    return similarities

def get_memory_usage():
    """Get current memory usage for available devices."""
    memory_stats = {}
    
    # CPU memory (rough estimate using process memory)
    import psutil
    process = psutil.Process()
    memory_stats['cpu'] = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # MPS memory (if available)
    if torch.backends.mps.is_available():
        # Note: MPS doesn't provide memory stats directly
        memory_stats['mps'] = "MPS memory stats not available"
        
    return memory_stats

def benchmark_implementations(n_vectors: int = 1000, vector_dim: int = 768):
    """Benchmark different implementations."""
    # Determine available device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running benchmarks on device: {device}")
    
    # Generate random test data
    torch.manual_seed(42)
    activations = [torch.randn(vector_dim, device=device) for _ in range(n_vectors)]
    
    implementations = [
        ('Original (triu_indices)', compute_pairwise_cosine_similarity_original),
        ('Nested Loops', compute_pairwise_cosine_similarity_loops),
        ('Boolean Mask', compute_pairwise_cosine_similarity_mask),
        ('Combinations', compute_pairwise_cosine_similarity_combinations),
        ('Einsum', compute_pairwise_cosine_similarity_einsum)
    ]
    
    results = {}
    for name, impl in implementations:
        print(f"\nTesting {name}...")
        
        # Warmup
        impl(activations[:10], device=device)
        
        # Actual timing
        start_time = time.perf_counter()
        result = impl(activations, device=device)
        end_time = time.perf_counter()
        
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        memory_stats = get_memory_usage()
        
        results[name] = {
            'time': elapsed_time,
            'memory': memory_stats,
            'result_size': len(result)
        }
        
        print(f"Time: {elapsed_time:.2f} ms")
        print(f"Memory usage: {memory_stats}")
    
    return results

# Example usage and comparison:
if __name__ == "__main__":
    # Small scale test for correctness
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_vectors = [
        torch.tensor([1., 2., 3.], device=device),
        torch.tensor([4., 5., 6.], device=device),
        torch.tensor([7., 8., 9.], device=device)
    ]
    
    # Verify all implementations give same results
    results = []
    for impl in [
        compute_pairwise_cosine_similarity_original,
        compute_pairwise_cosine_similarity_loops,
        compute_pairwise_cosine_similarity_mask,
        compute_pairwise_cosine_similarity_combinations,
        compute_pairwise_cosine_similarity_einsum
    ]:
        result = impl(test_vectors)
        results.append(result)
        print(f"{impl.__name__}: {result}")
        
    # Check all results are equal
    print("\nAll implementations equal:", 
          all(torch.allclose(torch.tensor(r1), torch.tensor(r2)) 
              for r1, r2 in zip(results[:-1], results[1:])))
    
    # Run benchmarks with different sizes
    print("\nRunning benchmarks with different sizes...")
    for size in [100, 500, 1000]:
        print(f"\nBenchmarking with {size} vectors...")
        benchmark_results = benchmark_implementations(n_vectors=size, vector_dim=768)