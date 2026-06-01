<img width="1227" height="142" alt="Screen Shot 2026-06-01 at 1 03 48 PM" src="https://github.com/user-attachments/assets/7def56cb-ebeb-4293-944c-f9ea49f248fe" />



<img width="713" height="376" alt="Screen Shot 2026-06-01 at 1 08 37 PM" src="https://github.com/user-attachments/assets/dc58d54a-ef75-4114-881d-c1b4ad16e20c" />
<img width="871" height="233" alt="Screen Shot 2026-06-01 at 1 11 15 PM" src="https://github.com/user-attachments/assets/5fb98c53-1a67-4910-ad09-1e82c308f10b" />

# Tensor Parallel Transformer + Distributed Embedding System (PyTorch + NCCL)

## Overview

I built a  **tensor-parallel distributed ML system using PyTorch and NCCL**, consisting of two connected components:

1. A tensor-parallel transformer inference engine
2. A distributed embedding pipeline built on top of the same GPU parallelism framework

The goal is to emulate how modern LLM systems combine:

- intra-layer model parallelism (tensor parallel transformer blocks)
- embedding generation pipelines
- distributed execution using NCCL collectives

This project demonstrates how both transformer computation and embedding workloads can be scaled across multiple GPUs using a unified distributed runtime.

---

# Part 1: Tensor Parallel Transformer Engine

## What I built

I implemented a simplified GPT-style transformer block where computation is explicitly split across GPUs using tensor parallelism.

---

## Architecture

### 1. Tensor-Parallel Attention

- Attention heads are sharded across GPUs
- Each GPU computes Q, K, V projections for its local heads
- Scaled dot-product attention is computed independently per rank
- Outputs are synchronized using `torch.distributed.all_reduce`

---

### 2. Tensor-Parallel MLP

- Feed-forward layers are split across GPUs (hidden dimension sharding)
- Each GPU computes a partial projection
- GELU activation is applied locally
- Outputs are aggregated using `all_reduce`

---

### 3. Distributed Execution

- Each process corresponds to one GPU (`torchrun`)
- NCCL backend handles inter-GPU communication
- Each rank executes identical model code over different tensor shards

---
## 4 Distributed Embedding Pipeline (run_embed.py)

## What it does

This module demonstrates a second distributed workload:

- Text input → tokenization → embedding lookup
- Tensor-parallel embedding projection
- Multi-GPU synchronization of embedding vectors

## Pipeline structure

Raw text  
→ Tokenizer  
→ Embedding layer  
→ Tensor-parallel projection  
→ NCCL synchronization  
→ Final embedding vector  
## Output

The system produces a consistent transformer forward pass:

```text
Output shape: torch.Size([B, T, 768])

