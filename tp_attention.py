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
    
        qkv = self.qkv(x)  # (B, T, 3D)
    
        # split into QKV first
        q, k, v = qkv.chunk(3, dim=-1)
    
        # now shard by heads (NOT by fake reshape)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
    
        # select local heads correctly
        start = self.rank * self.local_heads
        end = start + self.local_heads
    
        q = q[:, :, start:end]
        k = k[:, :, start:end]
        v = v[:, :, start:end]
    
        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim),
            dim=-1
        )
    
        out = attn @ v  # (B, T, local_heads, head_dim)
    
        out = out.reshape(B, T, -1)
    
        # IMPORTANT: no all_reduce needed here
        return self.out(out)
