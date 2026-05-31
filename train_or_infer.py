import os
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

    model = TinyTPTransformer(
        dim=512,
        heads=8,
        tp_size=world,
        rank=rank
    ).cuda()

    x = torch.randn(8, 32, 512).cuda()

    for _ in range(50):
        y = model(x)

    if rank == 0:
        print("Forward pass complete:", y.shape)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
