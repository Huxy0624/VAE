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
            # 保持和 baseline GroupNorm 同名参数，方便加载 checkpoint
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        x: (N, C, H, W_local)，宽度被 sequence-parallel 切分。
        策略：
          - 非分布式或 world_size=1：直接用 F.group_norm（和 baseline 完全一样）
          - 否则：
              1）all_gather 把所有 rank 的局部块拼成完整 x_full
              2）在 x_full 上调用一次 F.group_norm（全局统计）
              3）按宽度切回各自 rank 的 chunk
        这样得到的输出，和 baseline GroupNorm 在整张图上跑出来的结果数值完全一致。
        """
        # 非分布式 / 未初始化：退化为普通 GroupNorm
        if (not dist.is_available()) or (not dist.is_initialized()):
            return F.group_norm(
                x,
                self.num_groups,
                self.weight,
                self.bias,
                self.eps,
            )

        pg = self.process_group
        world_size = dist.get_world_size(pg)
        rank = dist.get_rank(pg)

        # 单卡时不用并行
        if world_size == 1:
            return F.group_norm(
                x,
                self.num_groups,
                self.weight,
                self.bias,
                self.eps,
            )

        N, C, H, W_local = x.shape
        assert C == self.num_channels

        # 1) all_gather：拿到完整宽度
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x, group=pg)
        x_full = torch.cat(x_list, dim=3)   # (N, C, H, W_global)

        # 2) 在完整特征图上做一次标准 GroupNorm
        y_full = F.group_norm(
            x_full,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
        )  # (N, C, H, W_global)

        # 3) 按宽度切回本 rank 的 chunk
        W_global = y_full.shape[3]
        assert W_global % world_size == 0, \
            f"Output width {W_global} must be divisible by world size {world_size}"

        W_chunk = W_global // world_size
        start = rank * W_chunk
        end = start + W_chunk

        y_local = y_full[..., start:end]    # (N, C, H, W_chunk)
        return y_local
