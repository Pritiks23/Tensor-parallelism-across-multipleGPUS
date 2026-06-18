"""This code takes tokenized text, runs it through a tensor-parallel transformer across multiple GPUs,
averages the output tokens into one vector, and projects it into a 256-dimensional embedding. """
import torch
import torch.nn as nn
import torch.distributed as dist
from tokenizer import encode
from tp_gpt2 import TPGPT2Block


class TPTextEmbedder(nn.Module):
    def __init__(self, dim, heads, world_size, rank):
        super().__init__()

        self.rank = rank
        self.world_size = world_size
        # This line creates a tensor-parallel GPT-style transformer block and moves it onto the GPU, meaning each GPU holds its own shard of the model and participates in distributed computation when the forward pass runs.
        self.backbone = TPGPT2Block(dim, heads, world_size, rank).cuda()

        # projection to embedding space (useful output)
        self.proj = nn.Linear(dim, 256).cuda()

    def forward(self, input_ids):
        # fake embedding lookup (minimal version)
        x = torch.randn(1, input_ids.shape[1], 768).cuda()

        x = self.backbone(x)

        # pool sequence → single vector
        x = x.mean(dim=1)

        return self.proj(x)
