import torch
import torch.distributed as dist

from distributed import setup
from tp_embedder import TPTextEmbedder
from tokenizer import encode


def main():
    rank, world = setup()

    model = TPTextEmbedder(
        dim=768,
        heads=12,
        world_size=world,
        rank=rank
    )

    # INPUT TEXTS (real use case)
    texts = [
        "Tensor parallelism distributes computation across GPUs",
        "Machine learning is transforming healthcare",
        "Distributed systems require synchronization"
    ]

    if rank == 0:
        inputs = [encode(t).cuda() for t in texts]
    else:
        inputs = None

    # broadcast structure
    dist.barrier()

    if rank == 0:
        print("\n=== EMBEDDING PIPELINE START ===")

    embeddings = []

    for i, t in enumerate(texts):
        input_ids = encode(t).cuda()

        emb = model(input_ids)

        embeddings.append(emb.detach().cpu())

        if rank == 0:
            print(f"Processed: {t[:40]}... → embedding shape {emb.shape}")

    if rank == 0:
        print("\n=== DONE ===")
        print("Output embedding vector size:", embeddings[0].shape)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
