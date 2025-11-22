"""
Simple standalone test for ParallelGroupNorm without distributed setup.
This tests the logic by manually simulating the distributed environment.
"""
import torch
from torch import nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_groupnorm_logic():
    """Test ParallelGroupNorm logic by comparing with standard GroupNorm."""
    print("Testing ParallelGroupNorm logic (non-distributed simulation)...")
    
    # Create a simple GroupNorm layer
    num_groups = 32
    num_channels = 128
    B, C, H, W = 2, num_channels, 32, 32
    
    # Create baseline norm
    baseline_norm = nn.GroupNorm(num_groups, num_channels)
    
    # Create input
    x_full = torch.randn(B, C, H, W)
    
    # Run baseline
    baseline_norm.eval()
    with torch.no_grad():
        out_baseline = baseline_norm(x_full)
    
    # Simulate width-partitioning with 4 ranks
    world_size = 4
    W_local = W // world_size
    
    # Process each "rank" separately using manual calculation
    outputs = []
    for rank in range(world_size):
        start_w = rank * W_local
        end_w = (rank + 1) * W_local
        x_chunk = x_full[:, :, :, start_w:end_w]
        
        # Manually compute what ParallelGroupNorm should do
        C_per_group = C // num_groups
        x_grouped = x_chunk.view(B, num_groups, C_per_group, H, W_local)
        
        # Compute local statistics
        local_sum = x_grouped.sum(dim=(2, 3, 4), keepdim=True)
        local_sq_sum = (x_grouped ** 2).sum(dim=(2, 3, 4), keepdim=True)
        
        # Store for aggregation
        if rank == 0:
            global_sum = local_sum.clone()
            global_sq_sum = local_sq_sum.clone()
        else:
            global_sum += local_sum
            global_sq_sum += local_sq_sum
    
    # Compute global statistics
    local_count = C_per_group * H * W_local
    global_count = local_count * world_size
    global_mean = global_sum / global_count
    global_var = global_sq_sum / global_count - global_mean ** 2
    eps = 1e-6
    
    # Normalize each chunk
    parallel_outputs = []
    for rank in range(world_size):
        start_w = rank * W_local
        end_w = (rank + 1) * W_local
        x_chunk = x_full[:, :, :, start_w:end_w]
        
        C_per_group = C // num_groups
        x_grouped = x_chunk.view(B, num_groups, C_per_group, H, W_local)
        
        x_normalized = (x_grouped - global_mean) / torch.sqrt(global_var + eps)
        x_normalized = x_normalized.view(B, C, H, W_local)
        
        # Apply affine transformation
        if baseline_norm.weight is not None:
            x_normalized = x_normalized * baseline_norm.weight.view(1, C, 1, 1)
        if baseline_norm.bias is not None:
            x_normalized = x_normalized + baseline_norm.bias.view(1, C, 1, 1)
        
        parallel_outputs.append(x_normalized)
    
    # Concatenate parallel outputs
    out_parallel = torch.cat(parallel_outputs, dim=3)
    
    # Compare
    max_diff = (out_parallel - out_baseline).abs().max().item()
    print(f"Max difference between parallel and baseline: {max_diff}")
    
    if torch.allclose(out_parallel, out_baseline, rtol=1e-3, atol=1e-5):
        print("✓ Logic test passed!")
        return True
    else:
        print("✗ Logic test failed!")
        print(f"  Baseline stats: mean={out_baseline.mean()}, std={out_baseline.std()}")
        print(f"  Parallel stats: mean={out_parallel.mean()}, std={out_parallel.std()}")
        return False

if __name__ == "__main__":
    success = test_groupnorm_logic()
    sys.exit(0 if success else 1)
