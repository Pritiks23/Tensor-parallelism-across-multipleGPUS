"""This TPMLP implements a tensor-parallel feedforward network where the large hidden dimension is split across GPUs, each GPU computes a shard of the MLP independently, and the results are combined using an all-reduce to reconstruct the full output. """
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class TPMLP(nn.Module):
    def __init__(self, dim, world_size, rank):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        self.hidden = 4 * dim
        assert self.hidden % world_size == 0

        self.local_hidden = self.hidden // world_size

        # Column-parallel projection (sharded output)
        self.w1 = nn.Linear(dim, self.local_hidden, bias=False)

        # Row-parallel projection (maps back to model dim)
        self.w2 = nn.Linear(self.local_hidden, dim, bias=False)

    def forward(self, x):
        # (B, T, D) -> (B, T, local_hidden)
        h = self.w1(x)
        h = F.gelu(h)

        # (B, T, D)
        out = self.w2(h)

        # IMPORTANT: reconstruct full tensor across GPUs
        dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return out
