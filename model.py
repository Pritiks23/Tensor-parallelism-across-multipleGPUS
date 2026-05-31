import torch.nn as nn
from tp_layers import TPLinear, TPAttention

class TinyTPTransformer(nn.Module):
    def __init__(self, dim, heads, tp_size, rank):
        super().__init__()

        self.attn = TPAttention(dim, heads, tp_size, rank)
        self.mlp = TPLinear(dim, 4 * dim, tp_size, rank)
        self.proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return self.proj(x)
