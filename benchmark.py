import time
import torch
import torch.distributed as dist

def benchmark(model, x, steps=100):
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(steps):
        model(x)

    torch.cuda.synchronize()
    end = time.time()

    tps = steps / (end - start)
    return tps
