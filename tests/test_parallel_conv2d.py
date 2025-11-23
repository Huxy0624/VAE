"""
Unit tests for ParallelConv2d.

Run with: torchrun --nproc_per_node=N tests/test_parallel_conv2d.py
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
from parallel_modules import ParallelConv2d


def test_parallel_conv2d(in_channels, out_channels, kernel_size, stride, padding):
    """Test basic ParallelConv2d functionality."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_group = dist.group.WORLD
    
    # Ensure same initialization and input across all ranks
    set_seed(42)
    
    # Create parallel and non-parallel conv layers
    parallel_conv = ParallelConv2d(
        in_channels, out_channels, kernel_size, stride, padding, process_group
    )
    baseline_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding
    )
    
    # Copy weights from parallel to baseline for fair comparison
    with torch.no_grad():
        baseline_conv.weight.copy_(parallel_conv.conv.weight)
        if baseline_conv.bias is not None:
            baseline_conv.bias.copy_(parallel_conv.conv.bias)

    if rank == 0:
        w_diff = (parallel_conv.conv.weight - baseline_conv.weight).abs().max().item()
        b_diff = (parallel_conv.conv.bias - baseline_conv.bias).abs().max().item()
        print("[DEBUG] weight max diff:", w_diff)
        print("[DEBUG] bias   max diff:", b_diff)
    
    # Create input
    B, C, H, W = 2, in_channels, 32, 32
    device = "cpu"  # Use CPU for compatibility
    x_full = torch.randn(B, C, H, W, device=device)
    
    # Split input along width dimension
    x_chunk, _ = split_along_width(x_full, world_size, rank)
    
    # Forward pass
    parallel_conv.eval()
    baseline_conv.eval()
    
    with torch.no_grad():
        out_parallel_chunk = parallel_conv(x_chunk)
        out_baseline = baseline_conv(x_full)
    
    # Gather parallel outputs along width
    out_parallel_list = [torch.zeros_like(out_parallel_chunk) for _ in range(world_size)]
    dist.all_gather(out_parallel_list, out_parallel_chunk, group=process_group)
    
    # Reconstruct full output
    if rank == 0:
        out_parallel_full = gather_along_width(out_parallel_list, out_baseline.shape[3])

        print("=== DEBUG CONFIG ===")
        print("in_channels:", in_channels,
              "out_channels:", out_channels,
              "kernel_size:", kernel_size,
              "stride:", stride,
              "padding:", padding)
        print("x_full shape:", x_full.shape)
        print("out_baseline shape:", out_baseline.shape)
        print("out_parallel_full shape:", out_parallel_full.shape)
        diff = (out_parallel_full - out_baseline).abs()
        print("max diff:", diff.max().item())
        print("mean diff:", diff.mean().item())

        if not torch.allclose(out_parallel_full, out_baseline, rtol=1e-3, atol=1e-5):
            raise AssertionError("Parallel and baseline outputs do not match")
        else:
            print(f"✓ ParallelConv2d test passed: in={in_channels}, out={out_channels}, "
                  f"kernel={kernel_size}, stride={stride}, padding={padding}")

    # Wait for all ranks to finish checking
    dist.barrier()


def run_tests():
    """Run all ParallelConv2d tests (assumes distributed is already set up)."""
    rank = dist.get_rank()
    if rank == 0:
        print("=" * 60)
        print("Running ParallelConv2d tests")
        print("=" * 60)
    
    # Test cases
    test_cases = [
        (32, 64, 3, 1, 1),
        (64, 64, 1, 1, 0),
        (128, 256, 3, 1, 1),
    ]
    
    for in_channels, out_channels, kernel_size, stride, padding in test_cases:
        test_parallel_conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    if rank == 0:
        print("=" * 60)
        print("All ParallelConv2d tests passed!")
        print("=" * 60)


def run_all_tests():
    """Run all ParallelConv2d tests (standalone, with setup/cleanup)."""
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
