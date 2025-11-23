from typing import Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F
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
        self.process_group = process_group or dist.group.WORLD
        
        # Store conv parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create the actual conv layer
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for ParallelConv2d using All-Gather strategy.
        
        Input x has shape (B, C, H, W_local).
        
        Strategy:
        1. All-gather input to get full feature map x_full
        2. Apply standard convolution on x_full
        3. Split output back to chunks
        """
        # Check if distributed is initialized and world_size > 1
        if not (dist.is_available() and dist.is_initialized()):
            return self.conv(x)
            
        world_size = dist.get_world_size(self.process_group)
        if world_size == 1:
            return self.conv(x)
            
        # 1. All-gather
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x, group=self.process_group)
        x_full = torch.cat(x_list, dim=3)
        
        # 2. Standard convolution
        out_full = self.conv(x_full)
        
        # 3. Split output
        W_global = out_full.shape[3]
        W_local = W_global // world_size
        rank = dist.get_rank(self.process_group)
        start = rank * W_local
        end = start + W_local
        
        return out_full[..., start:end]


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
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for ParallelGroupNorm using All-Gather strategy.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        world_size = dist.get_world_size(self.process_group)
        if world_size == 1:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        # 1. All-gather
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x, group=self.process_group)
        x_full = torch.cat(x_list, dim=3)

        # 2. Standard GroupNorm
        y_full = F.group_norm(
            x_full,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
        )

        # 3. Split output
        W_global = y_full.shape[3]
        W_local = W_global // world_size
        rank = dist.get_rank(self.process_group)
        start = rank * W_local
        end = start + W_local

        return y_full[..., start:end]


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
        Attention for width-partitioned input using All-Gather strategy.
        
        Input h_ has shape (B, C, H, W_local).
        
        Strategy:
        1. All-gather input to get full feature map h_full
        2. Apply standard attention on h_full
        3. Split output back to chunks
        """
        h_ = self.norm(h_)
        
        # Check if distributed is initialized and world_size > 1
        if not (dist.is_available() and dist.is_initialized()):
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)
            
            # Standard attention (baseline uses num_heads=1)
            B, C, H, W = q.shape
            q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
            k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
            v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
            h_attn = nn.functional.scaled_dot_product_attention(q, k, v)
            return rearrange(h_attn, "b 1 (h w) c -> b c h w", h=H, w=W)

        world_size = dist.get_world_size(self.process_group)
        if world_size == 1:
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)
            
            B, C, H, W = q.shape
            q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
            k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
            v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
            h_attn = nn.functional.scaled_dot_product_attention(q, k, v)
            return rearrange(h_attn, "b 1 (h w) c -> b c h w", h=H, w=W)

        # 1. All-gather
        h_list = [torch.empty_like(h_) for _ in range(world_size)]
        dist.all_gather(h_list, h_, group=self.process_group)
        h_full = torch.cat(h_list, dim=3)
        
        # 2. Standard attention on full input
        q = self.q(h_full)
        k = self.k(h_full)
        v = self.v(h_full)
        
        B, C, H, W_global = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        
        h_attn = nn.functional.scaled_dot_product_attention(q, k, v)
        h_attn = rearrange(h_attn, "b 1 (h w) c -> b c h w", h=H, w=W_global)
        
        # 3. Split output
        W_local = W_global // world_size
        rank = dist.get_rank(self.process_group)
        start = rank * W_local
        end = start + W_local
        
        return h_attn[..., start:end]
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        # Note: self.proj_out is a 1x1 conv, so we can apply it locally
        # But since attention output is split, we can just apply it locally
        # Wait, if we use all_gather for attention, the output is split.
        # proj_out is 1x1 conv, so it works on split input without communication.
        return x + self.proj_out(self.attention(x))


class ParallelUpsample(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        
        # Use standard Conv2d for the convolution after upsampling
        # This matches the baseline Upsample structure
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Upsample with width-partitioned input using All-Gather strategy.
        
        Input x has shape (B, C, H, W_local).
        
        Strategy:
        1. All-gather input to get full feature map x_full
        2. Apply interpolate (nearest neighbor) on x_full
        3. Apply standard convolution on upsampled full feature map
        4. Split output back to chunks
        """
        if not (dist.is_available() and dist.is_initialized()):
            x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            return self.conv(x)

        world_size = dist.get_world_size(self.process_group)
        if world_size == 1:
            x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            return self.conv(x)

        # 1. All-gather
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x, group=self.process_group)
        x_full = torch.cat(x_list, dim=3)
        
        # 2. Interpolate
        x_up_full = nn.functional.interpolate(x_full, scale_factor=2.0, mode="nearest")
        
        # 3. Standard convolution
        out_full = self.conv(x_up_full)
        
        # 4. Split output
        W_global = out_full.shape[3]
        W_local = W_global // world_size
        rank = dist.get_rank(self.process_group)
        start = rank * W_local
        end = start + W_local
        
        return out_full[..., start:end]


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

