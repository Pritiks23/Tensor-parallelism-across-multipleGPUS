import torch.nn as nn
from tp_layers import TPLinear, TPAttention


class TinyTPTransformer(nn.Module):
    def __init__(self, dim, num_heads, tp_size, rank):
        super().__init__()

        self.attn = TPAttention(dim, num_heads, tp_size, rank)

        # MLP expands 4x like GPT-style blocks
        self.mlp = TPLinear(dim, dim * 4, tp_size, rank)

        self.proj = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return self.proj(x)
