import torch
import torch.nn as nn
import torch.distributed as dist


# -----------------------------
# Tensor Parallel Linear (MLP)
# -----------------------------
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

        gathered = [torch.zeros_like(local) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local)

        return torch.cat(gathered, dim=-1)


# -----------------------------
# Tensor Parallel Attention
# -----------------------------
class TPAttention(nn.Module):
    def __init__(self, dim, num_heads, tp_size, rank):
        super().__init__()
        assert num_heads % tp_size == 0

        self.tp_size = tp_size
        self.rank = rank

        self.local_heads = num_heads // tp_size
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, -1, self.head_dim)
        k = k.view(B, T, -1, self.head_dim)
        v = v.view(B, T, -1, self.head_dim)

        start = self.rank * self.local_heads
        end = start + self.local_heads

        q = q[:, :, start:end]
        k = k[:, :, start:end]
        v = v[:, :, start:end]

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5),
            dim=-1
        )

        out = torch.matmul(attn, v)
        out = out.reshape(B, T, -1)

        dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return self.out(out)
