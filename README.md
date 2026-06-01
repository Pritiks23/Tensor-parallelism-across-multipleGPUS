<img width="1227" height="142" alt="Screen Shot 2026-06-01 at 1 03 48 PM" src="https://github.com/user-attachments/assets/7def56cb-ebeb-4293-944c-f9ea49f248fe" />



<img width="713" height="376" alt="Screen Shot 2026-06-01 at 1 08 37 PM" src="https://github.com/user-attachments/assets/dc58d54a-ef75-4114-881d-c1b4ad16e20c" />
<img width="871" height="233" alt="Screen Shot 2026-06-01 at 1 11 15 PM" src="https://github.com/user-attachments/assets/5fb98c53-1a67-4910-ad09-1e82c308f10b" />

# Tensor Parallel Transformer + Distributed Embedding System (PyTorch + NCCL)


# Tensor Parallelism Across Multiple GPUs

A PyTorch/NCCL project that demonstrates **intra-layer tensor parallelism** for a GPT-style block and reuses the same distributed runtime for a **multi-GPU embedding pipeline**.

---

## 1) What this project implements

This repository contains two distributed workloads running on the same process-group foundation:

1. **Tensor-parallel Transformer block** (`TPGPT2Block`)
   - Attention heads are sharded across ranks.
   - MLP hidden dimension is sharded across ranks.
   - Per-rank partial results are synchronized with collective communication.

2. **Distributed text embedding pipeline** (`TPTextEmbedder`)
   - Text is tokenized with GPT-2 tokenizer.
   - A tensor-parallel GPT-style backbone processes sequence features.
   - Sequence outputs are pooled and projected into a fixed embedding space.

This models a common distributed-systems pattern in modern ML infra: **local compute on shard-owned state + global synchronization to reconstruct logical outputs**.

---

## 2) Distributed systems design

### Process model
- One process per GPU via `torchrun`.
- `distributed.setup()` initializes `torch.distributed` with the `nccl` backend.
- `LOCAL_RANK` is used to pin each process to its GPU.

### Communication model
- **Collective primitive used:** `all_reduce(SUM)`
- Used after rank-local partial projections to produce globally consistent outputs.
- Synchronization barriers are used in embedding execution flow.

### Partitioning strategy
- **Attention:** head-wise partitioning (`local_heads = num_heads / world_size`)
- **MLP:** hidden dimension partitioning (`local_hidden = 4*dim / world_size`)
- Each rank executes identical code paths, but on shard-local slices.

---

## 3) Code architecture

- `distributed.py`
  - Process-group initialization (`nccl`), rank/world discovery, device assignment.

- `tp_attention.py`
  - Tensor-parallel attention module.
  - Shards heads per rank and performs rank-local attention compute.
  - Uses `dist.all_reduce` to aggregate projected output.

- `tp_mlp.py`
  - Tensor-parallel feed-forward block.
  - Column/row-style sharded projections.
  - Uses `dist.all_reduce` to reconstruct output contribution.

- `tp_gpt2.py`
  - GPT-style residual block: `LN -> TPAttention -> residual`, `LN -> TPMLP -> residual`.

- `tokenizer.py`
  - GPT-2 tokenizer wrapper (`encode`, `decode`).

- `tp_embedder.py`
  - Distributed embedding model built on tensor-parallel GPT block.
  - Mean pooling + projection to 256-d embedding vector.

- `run_embed.py`
  - End-to-end embedding pipeline driver across distributed ranks.

- `generate.py`
  - Transformer block execution driver for repeated forward passes.

---

## 4) Execution flow (embedding pipeline)

1. Initialize process group (`rank`, `world_size`, GPU binding).
2. Tokenize text input.
3. Build tensor-parallel backbone on each rank.
4. Run forward pass:
   - rank-local attention/MLP compute
   - global `all_reduce` synchronization
5. Pool sequence outputs to sentence-level representation.
6. Project to embedding space (256-d vector).
7. Rank 0 logs output status and shapes.

---

## 5) Prerequisites

- Linux with CUDA-capable GPUs
- PyTorch with NCCL support
- Python 3.9+
- `torchrun` available

Install dependencies:

```bash
cd <repo_root>
pip install -r requirements.txt
pip install transformers  # required by tokenizer.py
```

> `tokenizer.py` requires `transformers`. Keep it installed in addition to packages from `requirements.txt`.

---

## 6) How to run

### Run distributed embedding pipeline

```bash
cd <repo_root>
torchrun --standalone --nproc_per_node=2 run_embed.py
```

Expected rank-0 style output:

- embedding pipeline start/end logs
- per-text processing logs
- embedding shape prints (e.g., `[1, 256]`)

### Run tensor-parallel GPT block driver

```bash
cd <repo_root>
torchrun --standalone --nproc_per_node=2 generate.py
```

---

## 7) Why this is systems-relevant

This project maps directly to core distributed systems concerns used in production ML stacks:

- **Sharding strategy:** deterministic partition ownership by rank.
- **Collective synchronization:** all-rank consistency via NCCL collectives.
- **SPMD execution model:** identical program, disjoint data slices.
- **Scalability path:** increase `nproc_per_node` while preserving model dimension contracts.
- **Failure sensitivity:** correctness depends on process-group health and collective completion on every rank.

---

## 8) Current scope and limitations

- Educational/minimal implementation, not yet optimized for production throughput.
- No checkpoint loading or parameter initialization from pretrained GPT weights.
- No mixed precision, gradient checkpointing, or pipeline parallelism.
- Limited dependency pinning and validation automation.

---

## 9) Recommended next production-grade upgrades

1. Add proper tensor-parallel QKV/output projection sharding semantics aligned with Megatron-style layers.
2. Introduce profiling (`torch.profiler`, NCCL trace analysis) for comm/compute overlap.
3. Add robust launch/config management (world size validation, rank-safe logging, timeouts).
4. Add reproducibility controls (seeding per rank, deterministic modes where feasible).
5. Add tests for shape invariants, rank-consistency, and distributed correctness under multiple world sizes.

---

## 10) Summary

If your target is FAANG-grade distributed ML engineering interviews, this repo demonstrates the right fundamentals:

- tensor-parallel decomposition,
- NCCL-based collective synchronization,
- and composing model-parallel building blocks into a practical embedding workload.

It is a strong base for extending toward full-scale inference/training systems.
