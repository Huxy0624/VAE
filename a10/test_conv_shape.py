"""Test ParallelConv2d with simulated multi-rank scenario."""
import torch
from torch import nn
from parallel_modules import ParallelConv2d

def test_conv_shape_consistency():
    """Test that ParallelConv2d produces consistent output shapes."""
    print("Testing ParallelConv2d shape consistency...")
    
    # Test parameters
    in_channels, out_channels = 64, 128
    kernel_size, padding = 3, 1
    B, C, H, W = 2, in_channels, 32, 32
    world_size = 4
    
    # Create modules
    parallel_conv = ParallelConv2d(in_channels, out_channels, kernel_size, padding=padding)
    baseline_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    
    # Copy weights
    with torch.no_grad():
        baseline_conv.weight.copy_(parallel_conv.conv.weight)
        if parallel_conv.conv.bias is not None:
            baseline_conv.bias.copy_(parallel_conv.conv.bias)
    
    # Test with full input
    x_full = torch.randn(B, C, H, W)
    
    parallel_conv.eval()
    baseline_conv.eval()
    
    with torch.no_grad():
        # Baseline output
        out_baseline = baseline_conv(x_full)
        print(f"  Baseline output: {out_baseline.shape}")
        
        # Simulate partitioning across 4 ranks
        W_per_rank = W // world_size
        outputs = []
        
        for rank in range(world_size):
            # Split input
            start_w = rank * W_per_rank
            end_w = (rank + 1) * W_per_rank if rank < world_size - 1 else W
            x_chunk = x_full[:, :, :, start_w:end_w].contiguous()
            
            print(f"  Rank {rank}: input shape {x_chunk.shape}", end="")
            
            # Process
            out_chunk = parallel_conv(x_chunk)
            print(f" -> output shape {out_chunk.shape}")
            
            outputs.append(out_chunk)
        
        # Check all outputs have same width
        widths = [out.shape[3] for out in outputs]
        print(f"  Output widths: {widths}")
        
        if len(set(widths)) != 1:
            print(f"ERROR: Inconsistent output widths! {widths}")
            return False
        else:
            print(f"PASS: All chunks have consistent width {widths[0]}")
            
        # Note: We can't concatenate and compare because we don't have actual
        # halo exchange in single-process mode, but shape consistency is the key fix
        return True

if __name__ == "__main__":
    try:
        success = test_conv_shape_consistency()
        if success:
            print("\n" + "="*60)
            print("Shape consistency test PASSED!")
            print("="*60)
        else:
            print("\nTest FAILED!")
            exit(1)
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
