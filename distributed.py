import os
import torch
import torch.distributed as dist


def setup():
    """
    Initialize NCCL distributed environment (Vast.ai / torchrun compatible)
    """
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Correct GPU mapping for torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank
