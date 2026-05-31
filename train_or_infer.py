import torch
import torch.distributed as dist
from model import TinyTPTransformer


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world


def main():
    rank, world = setup()

    device = torch.device(f"cuda:{rank}")

    # 🔥 FIXED SAFE CONFIG
    dim = 512
    num_heads = 8   # must divide dim and be divisible by TP size

    model = TinyTPTransformer(
        dim=dim,
        num_heads=num_heads,
        tp_size=world,
        rank=rank
    ).to(device)

    x = torch.randn(8, 32, dim, device=device)

    # warmup
    for _ in range(10):
        y = model(x)

    torch.cuda.synchronize()

    # benchmark
    import time
    start = time.time()

    for _ in range(50):
        y = model(x)

    torch.cuda.synchronize()
    end = time.time()

    tps = 50 / (end - start)

    if rank == 0:
        print(f"\n✔ Output shape: {y.shape}")
        print(f"✔ Throughput: {tps:.2f} steps/sec")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
