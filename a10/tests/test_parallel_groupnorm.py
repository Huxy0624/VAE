"""
Unit tests for ParallelGroupNorm.

Run with: torchrun --nproc_per_node=N tests/test_parallel_groupnorm.py
"""

import os
# Set this before importing torch to avoid libuv issues on Windows
os.environ['USE_LIBUV'] = '0'

import torch
import torch.distributed as dist
from torch import nn
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import (
    setup_distributed, cleanup_distributed, set_seed,
    split_along_width, gather_along_width
)
from parallel_modules import ParallelGroupNorm


def test_parallel_groupnorm(num_groups, num_channels):
    """Test basic ParallelGroupNorm functionality."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_group = dist.group.WORLD
    
    set_seed(42 + rank)
    
    # Create parallel and non-parallel norm layers
    parallel_norm = ParallelGroupNorm(num_groups, num_channels, process_group=process_group)
    baseline_norm = nn.GroupNorm(num_groups, num_channels)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        if hasattr(parallel_norm, 'weight') and parallel_norm.weight is not None:
            baseline_norm.weight.copy_(parallel_norm.weight)
        if hasattr(parallel_norm, 'bias') and parallel_norm.bias is not None:
            baseline_norm.bias.copy_(parallel_norm.bias)
    
    # Create input
    B, C, H, W = 2, num_channels, 32, 32
    device = "cpu"  # Use CPU for compatibility
    x_full = torch.randn(B, C, H, W, device=device)
    
    # Split input along width dimension
    x_chunk, _ = split_along_width(x_full, world_size, rank)
    
    # Forward pass
    parallel_norm.eval()
    baseline_norm.eval()
    
    with torch.no_grad():
        out_parallel_chunk = parallel_norm(x_chunk)
        out_baseline = baseline_norm(x_full)
    
    # Gather parallel outputs for comparison
    out_parallel_list = [torch.zeros_like(out_parallel_chunk) for _ in range(world_size)]
    dist.all_gather(out_parallel_list, out_parallel_chunk, group=process_group)
    
    if rank == 0:
        # Reconstruct full output
        out_parallel_full = gather_along_width(out_parallel_list, out_baseline.shape[3])
        
        # Compare outputs using allclose
        if not torch.allclose(out_parallel_full, out_baseline, rtol=1e-3, atol=1e-5):
            print(f"ERROR: Parallel and baseline outputs do not match")
            print(f"  Max diff: {(out_parallel_full - out_baseline).abs().max().item()}")
            raise AssertionError("Parallel and baseline outputs do not match")
        else:
            print(f"✓ ParallelGroupNorm test passed: groups={num_groups}, channels={num_channels}")


def run_tests():
    """Run all ParallelGroupNorm tests (assumes distributed is already set up)."""
    rank = dist.get_rank()
    if rank == 0:
        print("=" * 60)
        print("Running ParallelGroupNorm tests")
        print("=" * 60)
    
    # Test cases
    test_cases = [
        (32, 128),
        (16, 256),
        (8, 512),
    ]
    
    for num_groups, num_channels in test_cases:
        test_parallel_groupnorm(num_groups, num_channels)
    
    if rank == 0:
        print("=" * 60)
        print("All ParallelGroupNorm tests passed!")
        print("=" * 60)


def run_all_tests():
    """Run all ParallelGroupNorm tests (standalone, with setup/cleanup)."""
    try:
        setup_distributed()
        run_tests()
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Rank {rank}: Test failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    run_all_tests()
