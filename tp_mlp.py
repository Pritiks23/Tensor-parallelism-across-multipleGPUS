import torch
import torch.nn as nn
import torch.distributed as dist


class TPMLP(nn.Module):
    def __init__(self, dim, world_size, rank):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        # expand dimension (typical transformer MLP expansion)
        self.hidden = 4 * dim
        self.local_hidden = self.hidden // world_size

        # Column-parallel first projection (sharded output)
        self.w1 = nn.Linear(dim, self.local_hidden, bias=False)

        # Row-parallel second projection (same input, sharded reduction)
        self.w2 = nn.Linear(self.local_hidden, dim, bias=False)

    def forward(self, x):
        # 1. each GPU computes its shard of expanded hidden state
        h_local = self.w1(x)

        # 2. activation applied locally (NO communication needed)
        h_local = torch.nn.functional.gelu(h_local)

        # 3. each GPU computes partial output
        out_local = self.w2(h_local)

        # 4. sum contributions across GPUs (THIS is the key step)
        dist.all_reduce(out_local, op=dist.ReduceOp.SUM)

        return out_local
