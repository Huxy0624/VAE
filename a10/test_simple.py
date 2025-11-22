"""Simple test without unicode."""
import torch
from torch import nn
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Don't initialize distributed
from parallel_modules import ParallelGroupNorm, ParallelConv2d

def test_modules():
    print("Testing ParallelGroupNorm...")
    norm = ParallelGroupNorm(32, 128)
    x = torch.randn(2, 128, 16, 16)
    norm.eval()
    with torch.no_grad():
        out = norm(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape
    print("PASS: ParallelGroupNorm")
    
    print("\nTesting ParallelConv2d...")
    conv = ParallelConv2d(64, 128, kernel_size=3, padding=1)
    x = torch.randn(2, 64, 16, 16)
    conv.eval()
    with torch.no_grad():
        out = conv(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    expected = torch.Size([2, 128, 16, 16])
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("PASS: ParallelConv2d")
    
    print("\nTesting consistency with baseline...")
    # Test GroupNorm
    parallel_norm = ParallelGroupNorm(32, 128)
    baseline_norm = nn.GroupNorm(32, 128)
    with torch.no_grad():
        if parallel_norm.weight is not None:
            baseline_norm.weight.copy_(parallel_norm.weight)
        if parallel_norm.bias is not None:
            baseline_norm.bias.copy_(parallel_norm.bias)
    
    x = torch.randn(2, 128, 16, 16)
    parallel_norm.eval()
    baseline_norm.eval()
    with torch.no_grad():
        out_parallel = parallel_norm(x)
        out_baseline = baseline_norm(x)
    
    max_diff = (out_parallel - out_baseline).abs().max().item()
    print(f"  GroupNorm max diff: {max_diff:.6e}")
    assert max_diff < 1e-4, f"Diff too large: {max_diff}"
    print("PASS: GroupNorm consistency")
    
    # Test Conv2d
    parallel_conv = ParallelConv2d(64, 128, kernel_size=3, padding=1)
    baseline_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    with torch.no_grad():
        baseline_conv.weight.copy_(parallel_conv.conv.weight)
        if parallel_conv.conv.bias is not None:
            baseline_conv.bias.copy_(parallel_conv.conv.bias)
    
    x = torch.randn(2, 64, 16, 16)
    parallel_conv.eval()
    baseline_conv.eval()
    with torch.no_grad():
        out_parallel = parallel_conv(x)
        out_baseline = baseline_conv(x)
    
    max_diff = (out_parallel - out_baseline).abs().max().item()
    print(f"  Conv2d max diff: {max_diff:.6e}")
    assert max_diff < 1e-5, f"Diff too large: {max_diff}"
    print("PASS: Conv2d consistency")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

if __name__ == "__main__":
    try:
        test_modules()
    except Exception as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
