import torch
import torch.nn as nn
import torch.distributed as dist

# ---------------------------
# Tensor Parallel Linear
# ---------------------------
class TPLinear(nn.Module):
    def __init__(self, in_f, out_f, tp_size, rank):
        super().__init__()
        assert out_f % tp_size == 0

        self.tp_size = tp_size
        self.rank = rank
        self.local_out = out_f // tp_size

        self.weight = nn.Parameter(torch.randn(self.local_out, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.local_out))

    def forward(self, x):
        local = torch.matmul(x, self.weight.t()) + self.bias

        # 🔥 NCCL ALL-GATHER (replaces concat)
        gathered = [torch.zeros_like(local) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local)

        return torch.cat(gathered, dim=-1)
