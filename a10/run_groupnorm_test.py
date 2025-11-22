"""
Wrapper to run test with environment variable set.
"""
import os
import sys

# Set environment variable before importing torch
os.environ['USE_LIBUV'] = '0'

# Now run the test
if __name__ == "__main__":
    # Import and run the test module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tests.test_parallel_groupnorm import run_all_tests
    run_all_tests()
