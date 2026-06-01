import torch
import torch.nn as nn
import torch.distributed as dist
import math


class TPAttention(nn.Module):
    def __init__(self, dim, num_heads, world_size, rank):
        super().__init__()

        assert dim % num_heads == 0
        assert num_heads % world_size == 0

        self.rank = rank
        self.world_size = world_size

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_heads = num_heads // world_size

        # Each GPU only stores its shard of QKV projection (TRUE TP)
        self.qkv = nn.Linear(dim, (3 * dim) // world_size, bias=False)

        self.out = nn.Linear(dim // world_size, dim // world_size, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x)

        # reshape local shard
        local_dim = D // self.world_size
        qkv = qkv.view(B, T, self.world_size, 3 * local_dim)

        qkv = qkv[:, :, self.rank, :]
        q, k, v = qkv.chunk(3, dim=-1)

        # heads per GPU
        q = q.view(B, T, self.local_heads, self.head_dim)
        k = k.view(B, T, self.local_heads, self.head_dim)
        v = v.view(B, T, self.local_heads, self.head_dim)

        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim),
            dim=-1
        )

        out = attn @ v  # (B, T, local_heads, head_dim)

        out = out.reshape(B, T, local_dim)

        # IMPORTANT: TP is disjoint → NO SUM REDUCTION HERE
        return self.out(out)
