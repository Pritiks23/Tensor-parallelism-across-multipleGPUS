""" 

This code defines one Transformer block where the input goes through two repeated stages:

Attention stage
Normalize input
Run tensor-parallel attention across GPUs
Add result back (residual connection)
MLP stage
Normalize again
Run tensor-parallel feedforward network across GPUs
Add result back again (residual connection)

---
 1. Why Attention + MLP both exist

A Transformer layer is built from two complementary operations:

Attention = “find relationships between tokens”

Lets each word look at other words.


🟢 MLP = “process each token individually”

After tokens have exchanged info, MLP transforms each token’s features.

It:

refines representation
adds nonlinearity
increases model capacity

So MLP handles:

feature transformation per token

"""
import torch.nn as nn
from tp_attention import TPAttention
from tp_mlp import TPMLP


class TPGPT2Block(nn.Module):
    def __init__(self, dim, heads, world_size, rank):
        super().__init__()

        self.attn = TPAttention(dim, heads, world_size, rank)
        self.mlp = TPMLP(dim, world_size, rank)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
