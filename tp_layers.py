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
        local_out = torch.matmul(x, self.weight.t()) + self.bias

        gathered = [torch.zeros_like(local_out) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_out)

        return torch.cat(gathered, dim=-1)


# -----------------------------
# Tensor Parallel Attention (FIXED)
# -----------------------------
class TPAttention(nn.Module):
    def __init__(self, dim, num_heads, tp_size, rank):
        super().__init__()

        # 🔥 CRITICAL FIX: enforce valid geometry
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.tp_size = tp_size
        self.rank = rank

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"

        self.local_heads = num_heads // tp_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x)

        # (B, T, 3D) → split
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, num_heads * head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # -----------------------------
        # Tensor Parallel head split
        # -----------------------------
        start = self.rank * self.local_heads
        end = start + self.local_heads

        q = q[:, :, start:end, :]
        k = k[:, :, start:end, :]
        v = v[:, :, start:end, :]

        # attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * scale,
            dim=-1
        )

        out = torch.matmul(attn, v)

        # merge heads back
        out = out.reshape(B, T, self.local_heads * self.head_dim)

        # -----------------------------
        # IMPORTANT: NCCL ALL-REDUCE
        # -----------------------------
        dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return self.out(out)
