"""
Custom distributed launcher that avoids libuv issues.
This manually sets up the distributed environment and runs tests.
"""
import os
import sys
import multiprocessing as mp

# Set environment variable before importing torch
os.environ['USE_LIBUV'] = '0'

import torch
import torch.distributed as dist


def run_worker(rank, world_size, test_module):
    """Run a single worker process."""
    # Set environment variables for this worker
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['USE_LIBUV'] = '0'
    
    # Initialize process group
    try:
        dist.init_process_group(
            backend='gloo',  # Use gloo backend for CPU
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        # Import and run the test
        if test_module == 'groupnorm':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from tests.test_parallel_groupnorm import run_tests
            run_tests()
        
        # Clean up
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Launch multiple worker processes."""
    world_size = 4
    test_module = sys.argv[1] if len(sys.argv) > 1 else 'groupnorm'
    
    print(f"Launching {world_size} workers to test {test_module}...")
    
    # Spawn worker processes
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(target=run_worker, args=(rank, world_size, test_module))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Check if all succeeded
    if all(p.exitcode == 0 for p in processes):
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                print(f"  Rank {i} failed with exit code {p.exitcode}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
