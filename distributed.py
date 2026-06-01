import torch
import torch.distributed as dist
import os


def setup():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return rank, world
