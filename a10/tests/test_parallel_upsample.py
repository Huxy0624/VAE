"""
Unit tests for ParallelUpsample.

Run with: torchrun --nproc_per_node=N tests/test_parallel_upsample.py
"""

import torch
import torch.distributed as dist
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import (
    setup_distributed, cleanup_distributed, set_seed,
    split_along_width, gather_along_width
)
from autoencoder_2d import Upsample
from parallel_modules import ParallelUpsample


def test_parallel_upsample(in_channels):
    """Test ParallelUpsample against baseline Upsample."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_group = dist.group.WORLD
    
    set_seed(42 + rank)
    
    # Create parallel and baseline upsample layers
    parallel_upsample = ParallelUpsample(in_channels, process_group=process_group)
    baseline_upsample = Upsample(in_channels)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        if hasattr(parallel_upsample, 'conv') and hasattr(baseline_upsample, 'conv'):
            baseline_upsample.conv.weight.copy_(parallel_upsample.conv.weight)
            if baseline_upsample.conv.bias is not None:
                baseline_upsample.conv.bias.copy_(parallel_upsample.conv.bias)
    
    # Create input
    B, C, H, W = 2, in_channels, 16, 16
    device = "cpu"  # Use CPU for compatibility
    x_full = torch.randn(B, C, H, W, device=device)
    
    # Split input along width dimension
    x_chunk, _ = split_along_width(x_full, world_size, rank)
    
    # Forward pass
    parallel_upsample.eval()
    baseline_upsample.eval()
    
    with torch.no_grad():
        out_parallel_chunk = parallel_upsample(x_chunk)
        out_baseline = baseline_upsample(x_full)
    
    # Gather parallel outputs
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
            print(f"✓ ParallelUpsample test passed: in_channels={in_channels}")


def run_tests():
    """Run all ParallelUpsample tests (assumes distributed is already set up)."""
    rank = dist.get_rank()
    if rank == 0:
        print("=" * 60)
        print("Running ParallelUpsample tests")
        print("=" * 60)
    
    # Test cases
    test_cases = [64, 128, 256]
    
    for in_channels in test_cases:
        test_parallel_upsample(in_channels)
    
    if rank == 0:
        print("=" * 60)
        print("All ParallelUpsample tests passed!")
        print("=" * 60)


def run_all_tests():
    """Run all ParallelUpsample tests (standalone, with setup/cleanup)."""
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
