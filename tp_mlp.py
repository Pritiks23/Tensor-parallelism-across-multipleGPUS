import torch
import torch.nn as nn
import torch.distributed as dist


class TPMLP(nn.Module):
    def __init__(self, dim, world_size, rank):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        self.hidden = 4 * dim
        self.local_hidden = self.hidden // world_size

        self.w1 = nn.Linear(dim, self.local_hidden, bias=False)
        self.w2 = nn.Linear(self.local_hidden, dim, bias=False)

    def forward(self, x):
        # shard projection
        x = self.w1(x)

        # combine across GPUs
        gathered = [torch.zeros_like(x) for _ in range(self.world_size)]
        dist.all_gather(gathered, x)

        x = torch.cat(gathered, dim=-1)

        return self.w2(x)
