from typing import Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import silu as swish

from autoencoder_2d import (
    AutoEncoderConfig,
    DiagonalGaussianDistribution
)


class ParallelConv2d(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD if dist.is_initialized() else None
        
        # Store conv parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create the actual conv layer (no padding, we'll handle it manually)
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=0  # We handle padding manually with halo exchange
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for ParallelConv2d.
        
        Input x has shape (B, C, H, W_local) where W_local is the width slice for this rank.
        
        Strategy:
        1. Apply top/bottom padding (normal padding, no communication needed)
        2. Exchange boundary regions with neighboring ranks (halo exchange)
        3. Concatenate halo regions to get padded input
        4. Apply convolution
        5. Handle output slicing based on stride
        """
        B, C, H, W_local = x.shape
        
        # Get rank and world_size
        rank = dist.get_rank(self.process_group) if dist.is_initialized() else 0
        world_size = dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        
        # Step 1: Apply padding to top and bottom (no communication needed)
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (0, 0, self.padding, self.padding), mode='constant', value=0)
            H = H + 2 * self.padding
        
        # Step 2: Halo exchange for left and right boundaries
        # We need to exchange `padding` pixels with neighbors
        if world_size > 1 and self.padding > 0:
            # Prepare halo regions to send
            left_halo = x[:, :, :, :self.padding].contiguous()  # Send to left neighbor
            right_halo = x[:, :, :, -self.padding:].contiguous()  # Send to right neighbor
            
            # Prepare buffers to receive
            recv_left = torch.zeros_like(left_halo)
            recv_right = torch.zeros_like(right_halo)
            
            # Determine left and right neighbors
            left_rank = rank - 1 if rank > 0 else None
            right_rank = rank + 1 if rank < world_size - 1 else None
            
            # Send/receive operations
            reqs = []
            
            # Send right halo to right neighbor, receive from left neighbor
            if right_rank is not None:
                req = dist.isend(right_halo, dst=right_rank, group=self.process_group)
                reqs.append(req)
            if left_rank is not None:
                req = dist.irecv(recv_left, src=left_rank, group=self.process_group)
                reqs.append(req)
            
            # Send left halo to left neighbor, receive from right neighbor
            if left_rank is not None:
                req = dist.isend(left_halo, dst=left_rank, group=self.process_group)
                reqs.append(req)
            if right_rank is not None:
                req = dist.irecv(recv_right, src=right_rank, group=self.process_group)
                reqs.append(req)
            
            # Wait for all communications to complete
            for req in reqs:
                req.wait()
            
            # Step 3: Concatenate halo regions
            # Add zero padding for missing neighbors to ensure consistent output size
            parts = []
            if left_rank is not None:
                parts.append(recv_left)
            else:
                # First rank: add zero padding on the left
                zero_padding = torch.zeros_like(left_halo)
                parts.append(zero_padding)
            
            parts.append(x)
            
            if right_rank is not None:
                parts.append(recv_right)
            else:
                # Last rank: add zero padding on the right
                zero_padding = torch.zeros_like(right_halo)
                parts.append(zero_padding)
            
            x_padded = torch.cat(parts, dim=3)
        elif self.padding > 0:
            # Single rank: add zero padding on left and right
            x_padded = torch.nn.functional.pad(x, (self.padding, self.padding, 0, 0), mode='constant', value=0)
        else:
            # No padding needed
            x_padded = x
        
        # Step 4: Apply convolution
        out = self.conv(x_padded)
        
        return out


class ParallelGroupNorm(nn.Module):
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        # Learnable parameters (same as nn.GroupNorm)
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for ParallelGroupNorm.
        
        Input x has shape (B, C, H, W_local) where W_local is the width slice for this rank.
        GroupNorm normalizes across (H, W) for each group of channels.
        Since input is width-partitioned, we need to compute statistics across all ranks.
        
        Strategy:
        1. Reshape to separate groups: (B, num_groups, C//num_groups, H, W_local)
        2. Compute local mean and variance across (H, W_local)
        3. Aggregate statistics across ranks using all_reduce
        4. Apply normalization with global statistics
        5. Apply affine transformation
        """
        B, C, H, W_local = x.shape
        assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"
        
        # Reshape to separate groups: (B, num_groups, C_per_group, H, W_local)
        C_per_group = C // self.num_groups
        x_grouped = x.view(B, self.num_groups, C_per_group, H, W_local)
        
        # Compute local statistics across spatial dimensions (H, W_local) and channels in group
        # Shape after reduction: (B, num_groups, 1, 1, 1)
        local_sum = x_grouped.sum(dim=(2, 3, 4), keepdim=True)
        local_sq_sum = (x_grouped ** 2).sum(dim=(2, 3, 4), keepdim=True)
        
        # Count local elements per group
        local_count = C_per_group * H * W_local
        
        # Aggregate across ranks
        world_size = dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        
        # All-reduce to get global sums
        global_sum = local_sum.clone()
        global_sq_sum = local_sq_sum.clone()
        if world_size > 1 and dist.is_initialized():
            dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=self.process_group)
            dist.all_reduce(global_sq_sum, op=dist.ReduceOp.SUM, group=self.process_group)
        
        # Compute global mean and variance
        global_count = local_count * world_size
        global_mean = global_sum / global_count
        global_var = global_sq_sum / global_count - global_mean ** 2
        
        # Normalize using global statistics
        x_normalized = (x_grouped - global_mean) / torch.sqrt(global_var + self.eps)
        
        # Reshape back to (B, C, H, W_local)
        x_normalized = x_normalized.view(B, C, H, W_local)
        
        # Apply affine transformation
        if self.weight is not None:
            x_normalized = x_normalized * self.weight.view(1, C, 1, 1)
        if self.bias is not None:
            x_normalized = x_normalized + self.bias.view(1, C, 1, 1)
        
        return x_normalized


class ParallelAttnBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD if dist.is_initialized() else None
        self.in_channels = in_channels
        
        # Create norm and projection layers (same as baseline)
        self.norm = ParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, process_group=process_group)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def attention(self, h_: Tensor) -> Tensor:
        """
        Ulysses attention for width-partitioned input.
        
        Input h_ has shape (B, C, H, W_local) where W_local is the width slice.
        
        Ulysses Strategy:
        1. Apply norm (handles width partitioning correctly)
        2. Compute Q, K, V projections
        3. Reshape to sequence format: (B, C, H, W_local) -> (B, num_heads, H*W_local, head_dim)
        4. Use all-to-all to switch from width partitioning to head partitioning
        5. Compute attention with local heads
        6. Use all-to-all to switch back to width partitioning
        7. Reshape back to spatial format
        """
        h_ = self.norm(h_)
        
        # Compute Q, K, V projections
        q = self.q(h_)  # (B, C, H, W_local)
        k = self.k(h_)  # (B, C, H, W_local)
        v = self.v(h_)  # (B, C, H, W_local)
        
        B, C, H, W_local = q.shape
        world_size = dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        
        # If world_size=1, just use standard attention (no communication needed)
        if world_size == 1:
            q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
            k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
            v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
            h_attn = nn.functional.scaled_dot_product_attention(q, k, v)
            return rearrange(h_attn, "b 1 (h w) c -> b c h w", h=H, w=W_local)
        
        # Reshape to sequence format: (B, C, H, W_local) -> (B, 1, H*W_local, C)
        # Note: baseline uses num_heads=1, so we follow the same
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        
        # Ulysses: Use all-to-all to redistribute from width-parallel to head-parallel
        # Currently: each rank has all heads for a slice of the sequence (width)
        # After all-to-all: each rank has a subset of heads for the full sequence
        
        # For Ulysses, we need to reshape to enable all-to-all communication
        # Shape: (B, num_heads, seq_len_local, head_dim) -> (B, num_heads, world_size, seq_len_local//world_size, head_dim)
        # But since num_heads=1 in baseline, we use C as the "head" dimension
        
        # Actually, let's reinterpret: treat each channel as a separate head
        # Reshape: (B, 1, H*W_local, C) -> (B, C, H*W_local)
        seq_len_local = H * W_local
        
        # Reshape for all-to-all: (B, num_heads=C, seq_local) -> (world_size, B, heads_local=C//world_size, seq_local)
        # Then all-to-all will give us: (world_size, B, heads_local, seq_local) -> (B, heads_local, world_size*seq_local)
        
        # Simpler approach: reshape to (B*C, H*W_local) for all-to-all, then back
        # But this doesn't work well with scaled_dot_product_attention
        
        # Let's use the proper Ulysses pattern:
        # Split heads across ranks, gather sequence
        num_heads = C  # Treat each channel as a head (head_dim=1)
        heads_per_rank = num_heads // self.world_size
        
        # Reshape: (B, 1, seq_local, C) -> (B, C, seq_local)
        q = q.squeeze(1).transpose(1, 2)  # (B, seq_local, C)
        k = k.squeeze(1).transpose(1, 2)  # (B, seq_local, C)
        v = v.squeeze(1).transpose(1, 2)  # (B, seq_local, C)
        
        # All-to-all to switch from width-parallel to head-parallel
        # Input: (B, seq_local, C) where seq is split, C is full
        # Output: (B, seq_full, C_local) where seq is full, C is split
        
        # Reshape for all-to-all: (B, seq_local, C) -> (B, seq_local, world_size, heads_per_rank)
        q = q.view(B, seq_len_local, world_size, heads_per_rank)
        k = k.view(B, seq_len_local, world_size, heads_per_rank)
        v = v.view(B, seq_len_local, world_size, heads_per_rank)
        
        # Transpose to prepare for all-to-all: (B, seq_local, world_size, heads_local) -> (world_size, B, seq_local, heads_local)
        q = q.permute(2, 0, 1, 3).contiguous()
        k = k.permute(2, 0, 1, 3).contiguous()
        v = v.permute(2, 0, 1, 3).contiguous()
        
        # Flatten for all_to_all
        q_shape = q.shape
        q_flat = q.view(world_size, -1)
        k_flat = k.view(world_size, -1)
        v_flat = v.view(world_size, -1)
        
        # All-to-all communication
        q_gathered = torch.zeros_like(q_flat)
        k_gathered = torch.zeros_like(k_flat)
        v_gathered = torch.zeros_like(v_flat)
        
        # Split and gather
        q_list = list(q_flat.chunk(world_size, dim=0))
        k_list = list(k_flat.chunk(world_size, dim=0))
        v_list = list(v_flat.chunk(world_size, dim=0))
        
        q_out_list = [torch.zeros_like(q_list[0]) for _ in range(world_size)]
        k_out_list = [torch.zeros_like(k_list[0]) for _ in range(world_size)]
        v_out_list = [torch.zeros_like(v_list[0]) for _ in range(world_size)]
        
        dist.all_to_all(q_out_list, q_list, group=self.process_group)
        dist.all_to_all(k_out_list, k_list, group=self.process_group)
        dist.all_to_all(v_out_list, v_list, group=self.process_group)
        
        q_gathered = torch.cat(q_out_list, dim=0)
        k_gathered = torch.cat(k_out_list, dim=0)
        v_gathered = torch.cat(v_out_list, dim=0)
        
        # Reshape back: (world_size, B, seq_local, heads_local) -> (B, heads_local, world_size*seq_local)
        q_gathered = q_gathered.view(world_size, B, seq_len_local, heads_per_rank)
        k_gathered = k_gathered.view(world_size, B, seq_len_local, heads_per_rank)
        v_gathered = v_gathered.view(world_size, B, seq_len_local, heads_per_rank)
        
        # Permute: (world_size, B, seq_local, heads_local) -> (B, heads_local, world_size*seq_local)
        q_gathered = q_gathered.permute(1, 3, 0, 2).contiguous().view(B, heads_per_rank, world_size * seq_len_local)
        k_gathered = k_gathered.permute(1, 3, 0, 2).contiguous().view(B, heads_per_rank, world_size * seq_len_local)
        v_gathered = v_gathered.permute(1, 3, 0, 2).contiguous().view(B, heads_per_rank, world_size * seq_len_local)
        
        # Reshape for attention: (B, heads_local, seq_full) -> (B, heads_local, seq_full, 1)
        q_attn = q_gathered.unsqueeze(-1).transpose(1, 2)  # (B, seq_full, heads_local, 1)
        k_attn = k_gathered.unsqueeze(-1).transpose(1, 2)  # (B, seq_full, heads_local, 1)
        v_attn = v_gathered.unsqueeze(-1).transpose(1, 2)  # (B, seq_full, heads_local, 1)
        
        # Reshape to (B, heads_local, seq_full, head_dim=1)
        q_attn = q_attn.transpose(1, 2)
        k_attn = k_attn.transpose(1, 2)
        v_attn = v_attn.transpose(1, 2)
        
        # Compute attention
        h_attn = nn.functional.scaled_dot_product_attention(q_attn, k_attn, v_attn)  # (B, heads_local, seq_full, 1)
        
        # Reshape: (B, heads_local, seq_full, 1) -> (B, heads_local, world_size*seq_local)
        h_attn = h_attn.squeeze(-1)  # (B, heads_local, seq_full)
        
        # Reverse all-to-all: switch from head-parallel back to width-parallel
        # Input: (B, heads_local, seq_full) -> need to reshape for all-to-all
        # Output: (B, C, seq_local)
        
        # Reshape: (B, heads_local, world_size*seq_local) -> (B, heads_local, world_size, seq_local)
        h_attn = h_attn.view(B, heads_per_rank, world_size, seq_len_local)
        
        # Permute: (B, heads_local, world_size, seq_local) -> (world_size, B, heads_local, seq_local)
        h_attn = h_attn.permute(2, 0, 1, 3).contiguous()
        
        # All-to-all
        h_flat = h_attn.view(world_size, -1)
        h_list = list(h_flat.chunk(world_size, dim=0))
        h_out_list = [torch.zeros_like(h_list[0]) for _ in range(world_size)]
        dist.all_to_all(h_out_list, h_list, group=self.process_group)
        h_gathered_back = torch.cat(h_out_list, dim=0)
        
        # Reshape: (world_size, B, seq_local, heads_local) -> (B, seq_local, C)
        h_gathered_back = h_gathered_back.view(world_size, B, seq_len_local, heads_per_rank)
        h_gathered_back = h_gathered_back.permute(1, 2, 0, 3).contiguous().view(B, seq_len_local, C)
        
        # Reshape back to spatial: (B, seq_local, C) -> (B, C, H, W_local)
        h_out = h_gathered_back.transpose(1, 2).contiguous()  # (B, C, seq_local)
        h_out = rearrange(h_out, "b c (h w) -> b c h w", h=H, w=W_local)
        
        return h_out
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        return x + self.proj_out(self.attention(x))


class ParallelUpsample(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        
        # Use ParallelConv2d for the convolution after upsampling
        self.conv = ParallelConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, process_group=process_group)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Upsample with width-partitioned input.
        
        Input x has shape (B, C, H, W_local).
        
        Strategy:
        1. Apply interpolate (nearest neighbor) to upsample - this works locally
        2. Apply ParallelConv2d which handles boundary communication
        """
        # Upsample using nearest neighbor interpolation (works locally)
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        
        # Apply convolution with halo exchange
        x = self.conv(x)
        
        return x


class ParallelResnetBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        # Use parallel versions of norm and conv
        self.norm1 = ParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, process_group=process_group)
        self.conv1 = ParallelConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, process_group=process_group)
        self.norm2 = ParallelGroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True, process_group=process_group)
        self.conv2 = ParallelConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, process_group=process_group)
        
        # Shortcut connection if channels change
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ParallelConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, process_group=process_group)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass matching baseline ResnetBlock."""
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        
        return x + h


class ParallelDecoder(nn.Module):
    
    def __init__(
        self,
        config: AutoEncoderConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, config.z_channels, curr_res, curr_res)
        
        # z to block_in (use parallel conv)
        self.conv_in = ParallelConv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1, process_group=process_group)
        
        # middle block
        self.mid = nn.Module()
        self.mid.block_1 = ParallelResnetBlock(in_channels=block_in, out_channels=block_in, process_group=process_group)
        self.mid.attn_1 = ParallelAttnBlock(block_in, process_group=process_group)
        self.mid.block_2 = ParallelResnetBlock(in_channels=block_in, out_channels=block_in, process_group=process_group)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ParallelResnetBlock(in_channels=block_in, out_channels=block_out, process_group=process_group))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = ParallelUpsample(block_in, process_group=process_group)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        
        # end
        self.norm_out = ParallelGroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True, process_group=process_group)
        self.conv_out = ParallelConv2d(block_in, config.out_ch, kernel_size=3, stride=1, padding=1, process_group=process_group)
    
    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass for ParallelDecoder.
        
        Input z has shape (B, C, H, W_local) - already width-partitioned.
        """
        # z to block_in
        h = self.conv_in(z)
        
        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.norm_out(h)
        h = swish(h)
        return self.conv_out(h)


class ParallelAutoEncoder(nn.Module):
    
    def __init__(
        self,
        config: AutoEncoderConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD if dist.is_initialized() else None
        
        from autoencoder_2d import Encoder
        
        # Use baseline encoder (no parallelism)
        self.encoder = Encoder(config)
        
        # Use parallel decoder
        self.decoder = ParallelDecoder(config, process_group=process_group)
        
        # Store config parameters
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor
        self.sample = config.sample
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent z with parallel decoder.
        
        Input z has shape (B, C, T, H, W) - full tensor on all ranks.
        We need to:
        1. Rearrange to (B*T, C, H, W)
        2. Partition along width dimension
        3. Decode with parallel decoder
        4. Gather results
        5. Rearrange back to (B, C, T, H, W)
        """
        T = z.shape[2]
        z = rearrange(z, "b c t h w -> (b t) c h w")
        z = z / self.scale_factor + self.shift_factor
        
        # Get rank and world_size
        rank = dist.get_rank(self.process_group) if dist.is_initialized() else 0
        world_size = dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        
        # Partition z along width dimension
        B_T, C, H, W = z.shape
        W_local = W // world_size
        remainder = W % world_size
        
        # Calculate start and end indices for this rank
        if rank < remainder:
            start_w = rank * (W_local + 1)
            end_w = start_w + W_local + 1
        else:
            start_w = remainder * (W_local + 1) + (rank - remainder) * W_local
            end_w = start_w + W_local
        
        # Slice along width
        z_chunk = z[:, :, :, start_w:end_w].contiguous()
        
        # Decode with parallel decoder
        x_chunk = self.decoder(z_chunk)
        
        # Gather results from all ranks
        if world_size > 1 and dist.is_initialized():
            x_list = [torch.zeros_like(x_chunk) for _ in range(world_size)]
            dist.all_gather(x_list, x_chunk, group=self.process_group)
            # Concatenate along width dimension
            x = torch.cat(x_list, dim=3)
        else:
            # Single rank, no gathering needed
            x = x_chunk
        
        # Rearrange back to (B, C, T, H, W)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=T)
        
        return x
    
    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, DiagonalGaussianDistribution, Tensor]:
        """
        Full forward pass: encode -> decode.
        
        Input x has shape (B, C, T, H, W).
        Encoding is done on full tensor (no parallelism).
        Decoding uses parallel decoder.
        """
        # Encode (no parallelism)
        T = x.shape[2]
        x_enc = rearrange(x, "b c t h w -> (b t) c h w")
        params = self.encoder(x_enc)
        params = rearrange(params, "(b t) c h w -> b c t h w", t=T)
        posterior = DiagonalGaussianDistribution(params)
        
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        z = self.scale_factor * (z - self.shift_factor)
        
        # Decode (with parallelism)
        x_rec = self.decode(z)
        
        return x_rec, posterior, z
    
    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight

