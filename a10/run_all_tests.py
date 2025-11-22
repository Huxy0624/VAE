"""
Main test runner that imports and runs all distributed tests.

This script sets up distributed once and runs all test modules sequentially.
This avoids the overhead of setting up distributed for each test.

Run with: torchrun --nproc_per_node=N run_all_tests.py
"""

import sys
import os
import torch.distributed as dist

# Add tests directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))

from tests.test_utils import setup_distributed, cleanup_distributed


def run_test_module(module_name, test_function_name="run_tests"):
    """
    Import and run a test module.
    
    Args:
        module_name: Name of the test module (e.g., "test_parallel_conv2d")
        test_function_name: Name of the function to run (default: "run_tests")
    """
    rank = dist.get_rank()
    
    try:
        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"Running {module_name}")
            print(f"{'=' * 70}\n")
        
        # Import the module
        module = __import__(f"tests.{module_name}", fromlist=[test_function_name])
        
        # Get the test function
        test_func = getattr(module, test_function_name)
        
        # Run the test (it will use the already-initialized distributed environment)
        test_func()
        
        if rank == 0:
            print(f"\n✓ {module_name} completed successfully\n")
        
    except Exception as e:
        if rank == 0:
            print(f"\n✗ {module_name} failed with error: {e}\n")
        raise


def main():
    """Main test runner."""
    rank = 0
    world_size = 1
    
    try:
        # Set up distributed once for all tests
        setup_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if rank == 0:
            print("=" * 70)
            print("Starting Distributed Test Suite")
            print(f"World size: {world_size}")
            print("=" * 70)
        
        # List of test modules to run (standalone tests only)
        test_modules = [
            "test_parallel_conv2d",
            "test_parallel_groupnorm",
            "test_parallel_upsample",
            "test_parallel_attn_block",
        ]
        
        # Run each test module (distributed is already set up)
        for module_name in test_modules:
            run_test_module(module_name)
        
        if rank == 0:
            print("=" * 70)
            print("All tests completed successfully!")
            print("=" * 70)
        
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Test suite failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

