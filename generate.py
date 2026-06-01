import torch
from distributed import setup
from tp_gpt2 import TPGPT2Block


def main():
    rank, world, local_rank = setup()

    dim = 768
    heads = 12

    model = TPGPT2Block(dim, heads, world, rank).cuda()

    x = torch.randn(2, 16, dim, device="cuda")

    with torch.no_grad():
        for _ in range(30):
            x = model(x)

    if rank == 0:
        print("✔ TP GPT-2 block executed successfully")
        print("✔ Output shape:", x.shape)


if __name__ == "__main__":
    main()
