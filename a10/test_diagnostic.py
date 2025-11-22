"""
Comprehensive single-process test to diagnose issues in parallel modules.
This tests each module in isolation without requiring distributed setup.
"""
import torch
from torch import nn
import torch.distributed as dist
import os
import sys

# Mock distributed environment for testing
class MockProcessGroup:
    pass

class MockDist:
    """Mock torch.distributed for single-process testing."""
    group = type('obj', (object,), {'WORLD': MockProcessGroup()})
    ReduceOp = type('obj', (object,), {'SUM': 'sum'})
    
    @staticmethod
    def get_rank(group=None):
        return int(os.environ.get('MOCK_RANK', 0))
    
    @staticmethod
    def get_world_size(group=None):
        return int(os.environ.get('MOCK_WORLD_SIZE', 1))
    
    @staticmethod
    def is_initialized():
        return True
    
    @staticmethod
    def all_reduce(tensor, op=None, group=None):
        # For single process, no-op (already has full data)
        pass
    
    @staticmethod
    def all_gather(tensor_list, tensor, group=None):
        # For single process, just copy
        for t in tensor_list:
            t.copy_(tensor)
    
    @staticmethod
    def all_to_all(output_list, input_list, group=None):
        # For single process, just copy
        for out, inp in zip(output_list, input_list):
            out.copy_(inp)
    
    @staticmethod
    def isend(tensor, dst, group=None):
        return type('obj', (object,), {'wait': lambda: None})()
    
    @staticmethod
    def irecv(tensor, src, group=None):
        return type('obj', (object,), {'wait': lambda: None})()

# Patch torch.distributed temporarily
original_dist = dist
sys.modules['torch.distributed'] = type(sys)('torch.distributed')
for attr in dir(MockDist):
    if not attr.startswith('_'):
        setattr(sys.modules['torch.distributed'], attr, getattr(MockDist, attr))

# Now import our modules
from parallel_modules import (
    ParallelGroupNorm, ParallelConv2d, ParallelAttnBlock, 
    ParallelUpsample, ParallelResnetBlock
)
from autoencoder_2d import AttnBlock, Upsample, ResnetBlock

# Restore original dist
sys.modules['torch.distributed'] = original_dist

def test_module_instantiation():
    """Test that all modules can be instantiated."""
    print("\n" + "="*60)
    print("Test 1: Module Instantiation")
    print("="*60)
    
    try:
        # Set mock environment
        os.environ['MOCK_RANK'] = '0'
        os.environ['MOCK_WORLD_SIZE'] = '1'
        
        norm = ParallelGroupNorm(32, 128)
        print("✓ ParallelGroupNorm instantiated")
        
        conv = ParallelConv2d(64, 128, kernel_size=3, padding=1)
        print("✓ ParallelConv2d instantiated")
        
        attn = ParallelAttnBlock(128)
        print("✓ ParallelAttnBlock instantiated")
        
        upsample = ParallelUpsample(64)
        print("✓ ParallelUpsample instantiated")
        
        resnet = ParallelResnetBlock(64, 128)
        print("✓ ParallelResnetBlock instantiated")
        
        print("\nAll modules instantiated successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test that forward passes work without errors."""
    print("\n" + "="*60)
    print("Test 2: Forward Pass (Single Process)")
    print("="*60)
    
    try:
        os.environ['MOCK_RANK'] = '0'
        os.environ['MOCK_WORLD_SIZE'] = '1'
        
        # Test ParallelGroupNorm
        norm = ParallelGroupNorm(32, 128)
        x = torch.randn(2, 128, 16, 16)
        norm.eval()
        with torch.no_grad():
            out = norm(x)
        assert out.shape == x.shape, f"GroupNorm shape mismatch: {out.shape} vs {x.shape}"
        print("✓ ParallelGroupNorm forward pass")
        
        # Test ParallelConv2d
        conv = ParallelConv2d(64, 128, kernel_size=3, padding=1)
        x = torch.randn(2, 64, 16, 16)
        conv.eval()
        with torch.no_grad():
            out = conv(x)
        expected_shape = torch.Size([2, 128, 16, 16])
        assert out.shape == expected_shape, f"Conv2d shape mismatch: {out.shape} vs {expected_shape}"
        print("✓ ParallelConv2d forward pass")
        
        # Test ParallelUpsample
        upsample = ParallelUpsample(64)
        x = torch.randn(2, 64, 16, 16)
        upsample.eval()
        with torch.no_grad():
            out = upsample(x)
        expected_shape = torch.Size([2, 64, 32, 32])
        assert out.shape == expected_shape, f"Upsample shape mismatch: {out.shape} vs {expected_shape}"
        print("✓ ParallelUpsample forward pass")
        
        # Test ParallelResnetBlock
        resnet = ParallelResnetBlock(64, 64)
        x = torch.randn(2, 64, 16, 16)
        resnet.eval()
        with torch.no_grad():
            out = resnet(x)
        assert out.shape == x.shape, f"ResnetBlock shape mismatch: {out.shape} vs {x.shape}"
        print("✓ ParallelResnetBlock forward pass")
        
        # Test ParallelAttnBlock
        attn = ParallelAttnBlock(128)
        x = torch.randn(2, 128, 16, 16)
        attn.eval()
        with torch.no_grad():
            out = attn(x)
        assert out.shape == x.shape, f"AttnBlock shape mismatch: {out.shape} vs {x.shape}"
        print("✓ ParallelAttnBlock forward pass")
        
        print("\nAll forward passes successful!")
        return True
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_consistency():
    """Test that outputs match baseline when world_size=1."""
    print("\n" + "="*60)
    print("Test 3: Output Consistency (world_size=1)")
    print("="*60)
    
    try:
        os.environ['MOCK_RANK'] = '0'
        os.environ['MOCK_WORLD_SIZE'] = '1'
        
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
        assert max_diff < 1e-4, f"GroupNorm diff too large: {max_diff}"
        print("✓ ParallelGroupNorm matches baseline")
        
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
        assert max_diff < 1e-5, f"Conv2d diff too large: {max_diff}"
        print("✓ ParallelConv2d matches baseline")
        
        # Test Upsample
        parallel_upsample = ParallelUpsample(64)
        baseline_upsample = Upsample(64)
        with torch.no_grad():
            baseline_upsample.conv.weight.copy_(parallel_upsample.conv.conv.weight)
            if parallel_upsample.conv.conv.bias is not None:
                baseline_upsample.conv.bias.copy_(parallel_upsample.conv.conv.bias)
        
        x = torch.randn(2, 64, 16, 16)
        parallel_upsample.eval()
        baseline_upsample.eval()
        with torch.no_grad():
            out_parallel = parallel_upsample(x)
            out_baseline = baseline_upsample(x)
        
        max_diff = (out_parallel - out_baseline).abs().max().item()
        print(f"  Upsample max diff: {max_diff:.6e}")
        assert max_diff < 1e-5, f"Upsample diff too large: {max_diff}"
        print("✓ ParallelUpsample matches baseline")
        
        print("\nAll consistency tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Diagnostic Test Suite for Parallel Modules")
    print("="*60)
    
    results = []
    results.append(("Instantiation", test_module_instantiation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Consistency", test_output_consistency()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    if all(r[1] for r in results):
        print("\n✓ All diagnostic tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. See output above for details.")
        sys.exit(1)
