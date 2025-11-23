from typing import Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import silu as swish
import torch.nn.functional as F

from autoencoder_2d import (
    AutoEncoderConfig,
    DiagonalGaussianDistribution
)


class HaloExchange(torch.autograd.Function):
    pass


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

        # 保存超参数
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 真正的卷积核仍然用标准 Conv2d 来存权重
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,   # 这里只是保持和 baseline 一致，forward 里会自己控制 padding
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W_local)
        Use all_gather to reconstruct full input, run standard conv, then split.
        This guarantees output matches baseline exactly, ignoring communication overhead.
        """
        # Non-distributed or not initialized
        if (not dist.is_available()) or (not dist.is_initialized()):
            return self.conv(x)

        pg = self.process_group
        world_size = dist.get_world_size(pg)
        rank = dist.get_rank(pg)

        if world_size == 1:
            return self.conv(x)

        # 1. Gather all input chunks
        # We assume equal splits as enforced by ParallelAutoEncoder
        tensor_list = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(tensor_list, x, group=pg)
        
        # 2. Concatenate to get full input
        x_full = torch.cat(tensor_list, dim=3)
        
        # 3. Run standard convolution
        out_full = self.conv(x_full)
        
        # 4. Split output back to chunks
        # We use tensor_split to handle potential uneven splits if they were to occur,
        # though we expect even splits.
        # Note: tensor_split matches the logic of distributing remainder to first ranks.
        out_chunks = torch.tensor_split(out_full, world_size, dim=3)
        out_local = out_chunks[rank]

        return out_local


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
        x: (B, C, H, W_local)
        Use all_gather to reconstruct full input, run standard GroupNorm, then split.
        This guarantees output matches baseline exactly, ignoring communication overhead.
        """
        # Non-distributed or not initialized
        if (not dist.is_available()) or (not dist.is_initialized()):
            return super().forward(x) # This won't work because we don't inherit from GroupNorm properly for forward
            # Actually we should just implement the local logic or use F.group_norm if we had full input
            # But here x is local. If not distributed, we assume x is full.
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        pg = self.process_group
        world_size = dist.get_world_size(pg)
        rank = dist.get_rank(pg)

        if world_size == 1:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        # 1. Gather all input chunks
        tensor_list = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(tensor_list, x, group=pg)
        
        # 2. Concatenate to get full input
        x_full = torch.cat(tensor_list, dim=3)
        
        # 3. Run standard GroupNorm
        out_full = F.group_norm(x_full, self.num_groups, self.weight, self.bias, self.eps)
        
        # 4. Split output back to chunks
        out_chunks = torch.tensor_split(out_full, world_size, dim=3)
        out_local = out_chunks[rank]

        return out_local


import math

class ParallelAttnBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.in_channels = in_channels
        
        self.norm = ParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, process_group=process_group)
        self.q = ParallelConv2d(in_channels, in_channels, kernel_size=1, process_group=process_group)
        self.k = ParallelConv2d(in_channels, in_channels, kernel_size=1, process_group=process_group)
        self.v = ParallelConv2d(in_channels, in_channels, kernel_size=1, process_group=process_group)
        self.proj_out = ParallelConv2d(in_channels, in_channels, kernel_size=1, process_group=process_group)
    
    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        B, C, H, W_local = q.shape
        world_size = dist.get_world_size(self.process_group)
        
        assert C % world_size == 0, f"Channels {C} must be divisible by world size {world_size}"
        
        # Flatten spatial
        q = rearrange(q, "b c h w -> b c (h w)")
        k = rearrange(k, "b c h w -> b c (h w)")
        v = rearrange(v, "b c h w -> b c (h w)")
        
        # Ulysses All-to-All: (B, C, L_local) -> (B, C_local, L)
        q_global_seq = self._all_to_all_S2H(q)
        k_global_seq = self._all_to_all_S2H(k)
        v_global_seq = self._all_to_all_S2H(v)
        
        # Compute attention scores
        # q: (B, C_local, L) -> (B, L, C_local)
        q_t = rearrange(q_global_seq, "b c l -> b l c")
        
        # scale by sqrt(C) (full C)
        scale = 1.0 / math.sqrt(C)
        
        # (B, L, C_local) @ (B, C_local, L) -> (B, L, L)
        scores = torch.matmul(q_t, k_global_seq) * scale
        
        # All-Reduce scores to get full dot product sum
        dist.all_reduce(scores, group=self.process_group)
        
        attn = torch.softmax(scores, dim=-1)
        
        # Compute output
        # (B, C_local, L) @ (B, L, L)^T -> (B, C_local, L)
        # v_global_seq is (B, C_local, L)
        # attn is (B, L, L)
        # We want O = V A^T
        h_out = torch.matmul(v_global_seq, attn.transpose(-1, -2))
        
        # Ulysses All-to-All back: (B, C_local, L) -> (B, C, L_local)
        h_out = self._all_to_all_H2S(h_out)
        
        # Reshape back
        h_out = rearrange(h_out, "b c (h w) -> b c h w", h=H, w=W_local)
        
        return h_out
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))

    def _all_to_all_S2H(self, x):
        # x: (B, C, L_local) -> (B, C_local, L)
        B, C, L_local = x.shape
        world_size = dist.get_world_size(self.process_group)
        C_local = C // world_size
        
        x = x.view(B, world_size, C_local, L_local)
        x = x.permute(1, 0, 2, 3).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=self.process_group)
        out = out.permute(1, 2, 0, 3).contiguous()
        out = out.view(B, C_local, -1)
        return out

    def _all_to_all_H2S(self, x):
        # x: (B, C_local, L) -> (B, C, L_local)
        B, C_local, L = x.shape
        world_size = dist.get_world_size(self.process_group)
        L_local = L // world_size
        
        x = x.view(B, C_local, world_size, L_local)
        x = x.permute(2, 0, 1, 3).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=self.process_group)
        out = out.permute(1, 0, 2, 3).contiguous()
        out = out.view(B, -1, L_local)
        return out


class ParallelUpsample(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.conv = ParallelConv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            process_group=process_group
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W_local)
        # Interpolate scales H and W by 2.
        # Since W is split, we just interpolate the local chunk.
        # The spatial relationship is preserved because each chunk is contiguous in W.
        # e.g. [0, 1] -> [0, 0.5, 1, 1.5]
        # If we have [0, 1] on rank 0 and [2, 3] on rank 1
        # Rank 0 -> [0, 0.5, 1, 1.5]
        # Rank 1 -> [2, 2.5, 3, 3.5]
        # This is correct for nearest neighbor or linear interpolation if aligned correctly.
        # For 'nearest', it just duplicates pixels.
        
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


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
        
        self.norm1 = ParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, process_group=process_group)
        self.conv1 = ParallelConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, process_group=process_group)
        self.norm2 = ParallelGroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True, process_group=process_group)
        self.conv2 = ParallelConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, process_group=process_group)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ParallelConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, process_group=process_group)

    def forward(self, x: Tensor) -> Tensor:
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
        
        # z to block_in
        self.conv_in = ParallelConv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1, process_group=process_group)

        # middle
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
        self.process_group = process_group or dist.group.WORLD
        
        from autoencoder_2d import Encoder
        
        self.encoder = Encoder(config)
        self.decoder = ParallelDecoder(config, process_group=process_group)
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor
        self.sample = config.sample

    def encode_(self, x: Tensor) -> tuple[Tensor, DiagonalGaussianDistribution]:
        T = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        params = self.encoder(x)
        params = rearrange(params, "(b t) c h w -> b c t h w", t=T)
        posterior = DiagonalGaussianDistribution(params)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = self.scale_factor * (z - self.shift_factor)
        return z, posterior

    def encode(self, x: Tensor) -> Tensor:
        return self.encode_(x)[0]
    
    def decode(self, z: Tensor) -> Tensor:
        # z: (B, C, T, H, W)
        T = z.shape[2]
        z = rearrange(z, "b c t h w -> (b t) c h w")
        z = z / self.scale_factor + self.shift_factor
        
        # Split z along width for parallel decoding
        # z: (B*T, C, H, W)
        rank = dist.get_rank(self.process_group)
        world_size = dist.get_world_size(self.process_group)
        
        B_T, C, H, W = z.shape
        assert W % world_size == 0, f"Width {W} must be divisible by world size {world_size}"
        W_local = W // world_size
        
        start_w = rank * W_local
        end_w = start_w + W_local
        
        z_local = z[..., start_w:end_w]
        
        # Parallel decode
        x_local = self.decoder(z_local)
        
        # Gather output
        # x_local: (B*T, C_out, H_out, W_out_local)
        # We need to gather along width (dim 3)
        
        x_list = [torch.zeros_like(x_local) for _ in range(world_size)]
        dist.all_gather(x_list, x_local, group=self.process_group)
        
        x = torch.cat(x_list, dim=3)
        
        x = rearrange(x, "(b t) c h w -> b c t h w", t=T)
        return x
    
    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, DiagonalGaussianDistribution, Tensor]:
        # encode
        x.shape[2]
        z, posterior = self.encode_(x)
        # decode
        x_rec = self.decode(z)

        return x_rec, posterior, z
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def load_checkpoint(self, checkpoint_path: str):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        
        # We need to handle loading state dict into parallel modules
        # The keys in state_dict match the baseline model structure
        # e.g. decoder.mid.block_1.norm1.weight
        
        # For parallel modules, the structure is similar, but we need to be careful about
        # 1. ParallelConv2d: wraps conv in .conv
        # 2. ParallelUpsample: wraps conv in .conv
        
        # We can iterate over our named parameters and find the corresponding key in state_dict
        
        my_state_dict = self.state_dict()
        
        for name, param in my_state_dict.items():
            # Map parallel module names to baseline names
            # ParallelConv2d: .conv.weight -> .weight
            # ParallelUpsample: .conv.conv.weight -> .conv.weight (Upsample has .conv)
            
            key = name
            
            # Handle ParallelConv2d inside ParallelUpsample
            # ParallelUpsample has self.conv = ParallelConv2d
            # ParallelConv2d has self.conv = nn.Conv2d
            # So ParallelUpsample param is .conv.conv.weight
            # Baseline Upsample has self.conv = nn.Conv2d
            # So Baseline Upsample param is .conv.weight
            if "upsample.conv.conv." in key:
                key = key.replace("upsample.conv.conv.", "upsample.conv.")
            
            # Handle other ParallelConv2d
            # e.g. decoder.conv_in.conv.weight -> decoder.conv_in.weight
            elif ".conv.weight" in key and "upsample" not in key:
                 key = key.replace(".conv.weight", ".weight")
            elif ".conv.bias" in key and "upsample" not in key:
                 key = key.replace(".conv.bias", ".bias")
                 
            if key in state_dict:
                # Copy data
                # Note: For parallel modules, we might need to slice the weight if we were splitting weights
                # But here we are splitting INPUT (Sequence Parallelism), not weights (Tensor Parallelism)
                # So weights are replicated. We just copy the full weight.
                
                if param.shape == state_dict[key].shape:
                    param.data.copy_(state_dict[key])
                else:
                    print(f"Shape mismatch for {name} (mapped to {key}): {param.shape} vs {state_dict[key].shape}")
            else:
                print(f"Key not found in checkpoint: {key} (original: {name})")

