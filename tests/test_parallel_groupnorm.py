"""
Unit tests for ParallelGroupNorm.

Run with: torchrun --nproc_per_node=N tests/test_parallel_groupnorm.py
"""

import torch
import torch.distributed as dist
from torch import nn
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import (
    setup_distributed, cleanup_distributed, set_seed,
    split_along_width, gather_along_width
)
from parallel_modules import ParallelGroupNorm


"""
Unit tests for ParallelGroupNorm.

Run with: torchrun --nproc_per_node=N tests/test_parallel_groupnorm.py
"""

import torch
import torch.distributed as dist
from torch import nn
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import (
    setup_distributed, cleanup_distributed, set_seed,
    split_along_width, gather_along_width
)
from parallel_modules import ParallelGroupNorm


def test_parallel_groupnorm(num_groups, num_channels, affine):
    """Test basic ParallelGroupNorm functionality."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_group = dist.group.WORLD
    
    set_seed(42 + rank)
    
    # Create parallel and non-parallel layers
    parallel_gn = ParallelGroupNorm(
        num_groups, num_channels, affine=affine, process_group=process_group
    )
    baseline_gn = nn.GroupNorm(
        num_groups, num_channels, affine=affine
    )
    
    # Copy weights
    if affine:
        with torch.no_grad():
            baseline_gn.weight.copy_(parallel_gn.weight)
            baseline_gn.bias.copy_(parallel_gn.bias)
    
    # Create input
    B, C, H, W = 2, num_channels, 32, 32
    device = "cpu"
    x_full = torch.randn(B, C, H, W, device=device)
    
    # Split input along width dimension
    x_chunk, _ = split_along_width(x_full, world_size, rank)
    
    # Forward pass
    parallel_gn.eval()
    baseline_gn.eval()
    
    with torch.no_grad():
        out_parallel_chunk = parallel_gn(x_chunk)
        out_baseline = baseline_gn(x_full)
    
    # Gather parallel outputs along width
    out_parallel_list = [torch.zeros_like(out_parallel_chunk) for _ in range(world_size)]
    dist.all_gather(out_parallel_list, out_parallel_chunk, group=process_group)
    
    # Reconstruct full output
    if rank == 0:
        out_parallel_full = gather_along_width(out_parallel_list, out_baseline.shape[3])
        
        # Compare outputs
        # GroupNorm can have slight numerical differences due to reduction order
        if not torch.allclose(out_parallel_full, out_baseline, rtol=1e-3, atol=1e-5):
            print(f"ERROR: Parallel and baseline outputs do not match")
            diff = (out_parallel_full - out_baseline).abs()
            print(f"  Max diff: {diff.max().item()}")
            print(f"  Mean diff: {diff.mean().item()}")
            raise AssertionError("Parallel and baseline outputs do not match")
        else:
            print(f"✓ ParallelGroupNorm test passed: groups={num_groups}, channels={num_channels}, affine={affine}")


def run_tests():
    """Run all ParallelGroupNorm tests."""
    rank = dist.get_rank()
    if rank == 0:
        print("=" * 60)
        print("Running ParallelGroupNorm tests")
        print("=" * 60)
    
    # Test cases
    test_cases = [
        (32, 64, True),
        (16, 32, False),
        (8, 128, True),
    ]
    
    for num_groups, num_channels, affine in test_cases:
        test_parallel_groupnorm(num_groups, num_channels, affine)
    
    if rank == 0:
        print("=" * 60)
        print("All ParallelGroupNorm tests passed!")
        print("=" * 60)


def run_all_tests():
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

