"""
Shared utilities for distributed testing (no pytest dependencies).
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment if not already initialized."""
    if not dist.is_initialized():
        # Use environment variables set by torchrun
        # Always use gloo backend for CPU
        backend = "gloo"
        
        # Get environment variables set by torchrun
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
    return dist.group.WORLD


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def split_along_width(x: torch.Tensor, world_size: int, rank: int):
    """
    Split tensor along width dimension for sequence parallelism.
    
    For a 4D tensor (B, C, H, W), splits along the W dimension.
    Each process gets a slice of width: (B, C, H, W_local)
    """
    B, C, H, W = x.shape
    
    # Split along width dimension
    chunk_size = W // world_size
    remainder = W % world_size
    
    # Distribute remainder across first few processes
    if rank < remainder:
        start_w = rank * (chunk_size + 1)
        end_w = start_w + chunk_size + 1
    else:
        start_w = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        end_w = start_w + chunk_size
    
    # Slice along width
    x_chunk = x[:, :, :, start_w:end_w]
    
    return x_chunk, (H, end_w - start_w)


def gather_along_width(x_chunks: list[torch.Tensor], original_W: int):
    """Gather width-parallel chunks back to full tensor."""
    # Concatenate along width dimension (dim=3)
    x_full = torch.cat(x_chunks, dim=3)
    return x_full


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


