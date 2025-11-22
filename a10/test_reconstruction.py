"""
Standalone script to generate reconstructed images using both baseline and parallel autoencoders.

This script supports three actions:
1. encode: Encode with baseline autoencoder and save latent code
2. baseline: Decode with baseline autoencoder (requires latent from encode action)
3. test: Decode with parallel autoencoder (requires latent from encode action)

Usage:
    # Step 1: Encode image and save latent
    python test_reconstruction.py encode

    # Step 2: Decode with baseline
    python test_reconstruction.py baseline

    # Step 3: Decode with parallel implementation (requires distributed execution)
    torchrun --nproc_per_node=N python test_reconstruction.py test
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from PIL import Image
import torchvision.transforms as transforms
from autoencoder_2d import AutoEncoder, AutoEncoderConfig

# Try to import parallel modules (may not be fully implemented)
try:
    from parallel_modules import ParallelAutoEncoder
    PARALLEL_AVAILABLE = True
except (ImportError, NotImplementedError):
    PARALLEL_AVAILABLE = False
    print("Warning: ParallelAutoEncoder not available, will only test baseline")

# Configuration - mapped from Diffusers AutoencoderKL config
# Diffusers config: latent_channels=16, sample_size=1024, scaling_factor=0.3611, shift_factor=0.1159
# block_out_channels=[128, 256, 512, 512] → ch_mult=[1, 2, 4, 4] with ch=128
CONFIG = AutoEncoderConfig(
    from_pretrained=None,
    cache_dir=None,
    resolution=1024,  # sample_size from Diffusers config
    in_channels=3,
    ch=128,  # base channels (block_out_channels[0])
    out_ch=3,
    ch_mult=[1, 2, 4, 4],  # block_out_channels=[128,256,512,512] relative to ch=128
    num_res_blocks=2,  # layers_per_block from Diffusers config
    z_channels=16,  # latent_channels from Diffusers config
    scale_factor=0.3611,  # scaling_factor from Diffusers config
    shift_factor=0.1159,  # shift_factor from Diffusers config
    sample=False,  # Use deterministic mode (mean) instead of sampling
)

CHECKPOINT_PATH = "misc/ae.safetensors"
IMAGE_PATH = "misc/sample.jpeg"
LATENT_PATH = "latent_code.pt"
OUTPUT_BASELINE_PATH = "reconstructed_baseline.jpg"
OUTPUT_PARALLEL_PATH = "reconstructed_parallel.jpg"


def setup_distributed():
    """Initialize distributed training if running with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        
        # Initialize process group (always use gloo backend for CPU)
        dist.init_process_group(backend="gloo")
        
        # Set device (always use CPU)
        device = torch.device("cpu")
        
        return True, rank, world_size, device
    else:
        return False, 0, 1, None


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_and_preprocess_image(image_path: str, target_size: int = 512):
    """
    Load and preprocess an image for the autoencoder.
    
    Args:
        image_path: Path to the input image
        target_size: Target resolution (should match config.resolution)
    
    Returns:
        Preprocessed image tensor of shape (1, 3, 1, H, W) - (batch, channels, time, height, width)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Resize to target resolution
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),  # Converts to [0, 1] range and (C, H, W)
    ])
    
    img_tensor = transform(img)  # Shape: (C, H, W) in [0, 1]
    
    # Normalize to [-1, 1] range (common for VAE models)
    img_tensor = img_tensor * 2.0 - 1.0
    
    # Add batch and time dimensions: (C, H, W) -> (1, C, 1, H, W)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
    
    return img_tensor


def save_latent(latent: torch.Tensor, latent_path: str):
    """
    Save latent code to disk.
    
    Args:
        latent: Latent tensor
        latent_path: Path to save the latent code
    """
    torch.save(latent.cpu(), latent_path)
    print(f"Latent code saved to: {latent_path}")
    print(f"Latent code shape: {latent.shape}")


def load_latent(latent_path: str):
    """
    Load latent code from disk.
    
    Args:
        latent_path: Path to the latent code file
    
    Returns:
        Latent tensor
    """
    if not os.path.exists(latent_path):
        raise FileNotFoundError(
            f"Latent code file not found: {latent_path}\n"
            f"Please run 'python test_reconstruction.py encode' first to generate the latent code."
        )
    latent = torch.load(latent_path)
    print(f"Latent code loaded from: {latent_path}")
    print(f"Latent code shape: {latent.shape}")
    return latent


def save_reconstructed_image(tensor: torch.Tensor, output_path: str):
    """
    Save a reconstructed image tensor to disk.
    
    Args:
        tensor: Image tensor in [-1, 1] range, shape can be (1, 3, 1, H, W) or (1, 3, H, W) or (3, H, W)
        output_path: Path to save the image
    """
    # Remove all dimensions of size 1 to get (C, H, W) format
    img_tensor = tensor.squeeze()
    
    # If still more than 3 dimensions, take the first 3
    if img_tensor.ndim > 3:
        # Take the first slice along extra dimensions
        while img_tensor.ndim > 3:
            img_tensor = img_tensor[0]
    
    # Ensure we have (C, H, W) format
    if img_tensor.ndim == 2:
        # If we have (H, W), add channel dimension (grayscale)
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.ndim != 3:
        raise ValueError(
            f"Expected tensor with 2 or 3 dimensions after squeezing, "
            f"got {img_tensor.ndim} dimensions. Original shape: {tensor.shape}"
        )
    
    # Denormalize from [-1, 1] to [0, 1]
    img_tensor = (img_tensor + 1.0) / 2.0
    
    # Clamp to valid range
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    
    # Convert to PIL Image and save
    to_pil = transforms.ToPILImage()
    img = to_pil(img_tensor)
    img.save(output_path)
    print(f"Reconstructed image saved to: {output_path}")


def action_encode(device):
    """Action 1: Encode with baseline autoencoder and save latent."""
    print("\n" + "="*60)
    print("Action: Encode with Baseline AutoEncoder")
    print("="*60)
    
    # Create model
    model = AutoEncoder(CONFIG)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    model.load_checkpoint(CHECKPOINT_PATH)
    print("Checkpoint loaded successfully!")
    
    # Load and preprocess image
    print(f"Loading image from: {IMAGE_PATH}")
    img_tensor = load_and_preprocess_image(IMAGE_PATH, target_size=CONFIG.resolution)
    print(f"Image shape: {img_tensor.shape}")
    
    # Move to device
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    
    # Encode
    print("Encoding image...")
    with torch.no_grad():
        z = model.encode(img_tensor)
        print(f"Latent code shape: {z.shape}")
    
    # Save latent code
    save_latent(z, LATENT_PATH)
    
    print("\n" + "="*60)
    print("Encoding completed!")
    print("="*60)


def action_baseline(device):
    """Action 2: Decode with baseline autoencoder."""
    print("\n" + "="*60)
    print("Action: Decode with Baseline AutoEncoder")
    print("="*60)
    
    # Create model
    model = AutoEncoder(CONFIG)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    model.load_checkpoint(CHECKPOINT_PATH)
    print("Checkpoint loaded successfully!")
    
    # Load latent code
    z = load_latent(LATENT_PATH)
    
    # Move to device
    model = model.to(device)
    z = z.to(device)
    
    # Decode
    print("Decoding image...")
    with torch.no_grad():
        reconstructed = model.decode(z)
        print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # Move back to CPU for saving
    reconstructed = reconstructed.cpu()
    
    # Save reconstructed image
    save_reconstructed_image(reconstructed, OUTPUT_BASELINE_PATH)
    
    print("\n" + "="*60)
    print("Baseline decoding completed!")
    print("="*60)


def action_test(device, rank, world_size, process_group):
    """Action 3: Decode with parallel autoencoder."""
    if not PARALLEL_AVAILABLE:
        print("\n" + "="*60)
        print("ParallelAutoEncoder not available, cannot run test action")
        print("="*60)
        raise RuntimeError("ParallelAutoEncoder not available")
    
    print("\n" + "="*60)
    print(f"Action: Decode with Parallel AutoEncoder (rank {rank}/{world_size})")
    print("="*60)
    
    # Create model
    model = ParallelAutoEncoder(CONFIG, process_group=process_group)
    model.eval()
    
    # Load checkpoint (only rank 0 loads, then broadcast)
    if rank == 0:
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        model.load_checkpoint(CHECKPOINT_PATH)
        print("Checkpoint loaded successfully!")
    
    # Broadcast model state from rank 0 to all ranks
    # Note: This is simplified - in practice you'd need to properly broadcast all parameters
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.data, src=0, group=process_group)
    
    # Load latent code (every rank loads from file)
    z_full = load_latent(LATENT_PATH)
    if rank == 0:
        print(f"Latent code shape: {z_full.shape}")
    
    # Move to device
    model = model.to(device)
    z_full = z_full.to(device)
    
    # Decode
    print(f"[Rank {rank}] Decoding image...")
    with torch.no_grad():
        reconstructed = model.decode(z_full)
        if rank == 0:
            print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # Gather reconstructed outputs from all ranks (if needed)
    # For now, assume rank 0 has the full output
    if rank == 0:
        reconstructed = reconstructed.cpu()
        
        # Save reconstructed image
        save_reconstructed_image(reconstructed, OUTPUT_PARALLEL_PATH)
        
        print("\n" + "="*60)
        print("Parallel decoding completed!")
        print("="*60)


def main():
    """Main function to execute the specified action."""
    parser = argparse.ArgumentParser(
        description="Test reconstruction with baseline and parallel autoencoders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Actions:
  encode    Encode image with baseline autoencoder and save latent code
  baseline  Decode latent code with baseline autoencoder
  test      Decode latent code with parallel autoencoder (requires distributed execution)

Examples:
  python test_reconstruction.py encode
  python test_reconstruction.py baseline
  torchrun --nproc_per_node=4 python test_reconstruction.py test
        """
    )
    parser.add_argument(
        "action",
        choices=["encode", "baseline", "test"],
        help="Action to perform: encode, baseline, or test"
    )
    
    args = parser.parse_args()
    
    # Setup distributed if running with torchrun
    is_distributed, rank, world_size, device = setup_distributed()
    
    if device is None:
        # Single process mode - always use CPU
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    if is_distributed:
        print(f"Running in distributed mode: rank {rank}/{world_size}")
        process_group = dist.group.WORLD
    else:
        print("Running in single-process mode")
        process_group = None
    
    try:
        if args.action == "encode":
            if is_distributed:
                print("Warning: encode action does not require distributed execution")
            action_encode(device)
        
        elif args.action == "baseline":
            if is_distributed:
                print("Warning: baseline action does not require distributed execution")
            action_baseline(device)
        
        elif args.action == "test":
            if not is_distributed:
                print("\nError: test action requires distributed execution")
                print("Run with: torchrun --nproc_per_node=N python test_reconstruction.py test")
                sys.exit(1)
            action_test(device, rank, world_size, process_group)
    
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

