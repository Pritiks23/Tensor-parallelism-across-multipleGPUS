import torch
from distributed import setup
from tp_gpt2 import TPGPT2Block


def main():
    rank, world = setup()

    dim = 768
    heads = 12

    model = TPGPT2Block(dim, heads, world, rank).cuda()

    x = torch.randn(2, 16, dim).cuda()

    for _ in range(30):
        x = model(x)

    if rank == 0:
        print("✔ TP GPT-2 block executed")
        print("✔ Output shape:", x.shape)


if __name__ == "__main__":
    main()
