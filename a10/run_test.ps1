# Set environment variable to disable libuv
$env:USE_LIBUV = '0'

# Run torchrun with arguments
torchrun --nproc_per_node=4 $args
