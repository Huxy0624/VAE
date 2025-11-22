"""
Unit tests for ParallelAttnBlock (Ulysses attention).

Run with: torchrun --nproc_per_node=N tests/test_parallel_attn_block.py
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
from autoencoder_2d import AttnBlock
from parallel_modules import ParallelAttnBlock


def test_parallel_attn_block(in_channels):
    """Test ParallelAttnBlock against baseline AttnBlock."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_group = dist.group.WORLD
    
    set_seed(42 + rank)
    
    # Create parallel and baseline attention blocks
    parallel_attn = ParallelAttnBlock(in_channels, process_group=process_group)
    baseline_attn = AttnBlock(in_channels)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        # Copy norm weights
        if hasattr(parallel_attn, 'norm') and hasattr(baseline_attn, 'norm'):
            if hasattr(parallel_attn.norm, 'weight') and parallel_attn.norm.weight is not None:
                baseline_attn.norm.weight.copy_(parallel_attn.norm.weight)
            if hasattr(parallel_attn.norm, 'bias') and parallel_attn.norm.bias is not None:
                baseline_attn.norm.bias.copy_(parallel_attn.norm.bias)
        
        # Copy conv weights (q, k, v, proj_out)
        for name in ['q', 'k', 'v', 'proj_out']:
            if hasattr(parallel_attn, name) and hasattr(baseline_attn, name):
                parallel_conv = getattr(parallel_attn, name)
                baseline_conv = getattr(baseline_attn, name)
                if hasattr(parallel_conv, 'weight'):
                    baseline_conv.weight.copy_(parallel_conv.weight)
                if hasattr(parallel_conv, 'bias') and parallel_conv.bias is not None:
                    baseline_conv.bias.copy_(parallel_conv.bias)
    
    # Create input
    B, C, H, W = 2, in_channels, 32, 32
    device = "cpu"  # Use CPU for compatibility
    x_full = torch.randn(B, C, H, W, device=device)
    
    # Split input along width dimension
    x_chunk, _ = split_along_width(x_full, world_size, rank)
    
    # Forward pass
    parallel_attn.eval()
    baseline_attn.eval()
    
    with torch.no_grad():
        out_parallel_chunk = parallel_attn(x_chunk)
        out_baseline = baseline_attn(x_full)
    
    # Gather parallel outputs along width
    out_parallel_list = [torch.zeros_like(out_parallel_chunk) for _ in range(world_size)]
    dist.all_gather(out_parallel_list, out_parallel_chunk, group=process_group)
    
    if rank == 0:
        # Reconstruct full output by concatenating along width dimension
        out_parallel_full = gather_along_width(out_parallel_list, out_baseline.shape[3])
        
        # Compare outputs using allclose
        if not torch.allclose(out_parallel_full, out_baseline, rtol=1e-3, atol=1e-5):
            print(f"ERROR: Parallel and baseline outputs do not match")
            print(f"  Max diff: {(out_parallel_full - out_baseline).abs().max().item()}")
            raise AssertionError("Parallel and baseline outputs do not match")
        else:
            print(f"✓ ParallelAttnBlock test passed: in_channels={in_channels}")


def run_tests():
    """Run all ParallelAttnBlock tests (assumes distributed is already set up)."""
    rank = dist.get_rank()
    if rank == 0:
        print("=" * 60)
        print("Running ParallelAttnBlock tests")
        print("=" * 60)
    
    # Test cases
    test_cases = [64, 128, 256]
    
    for in_channels in test_cases:
        test_parallel_attn_block(in_channels)
    
    if rank == 0:
        print("=" * 60)
        print("All ParallelAttnBlock tests passed!")
        print("=" * 60)


def run_all_tests():
    """Run all ParallelAttnBlock tests (standalone, with setup/cleanup)."""
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
