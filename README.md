Tensor Parallel Mini GPT (PyTorch + NCCL)
Overview

This project is a minimal, from-scratch implementation of tensor parallelism for transformer models using PyTorch and NCCL. It demonstrates how large model computations can be split across multiple GPUs and synchronized using distributed communication primitives.

The goal is to replicate the core ideas behind large-scale model parallel systems (such as Megatron-style architectures) in a simplified, runnable form suitable for multi-GPU environments like Vast.ai.

What this project does

This system builds a tiny transformer block and distributes its computation across multiple GPUs using tensor parallelism:

1. Tensor Parallel MLP (Column Parallelism)
The weight matrix in the feed-forward layer is split across GPUs.
Each GPU computes a partial output independently.
Outputs are concatenated across devices using NCCL communication.
2. Tensor Parallel Attention (Head Parallelism)
Attention heads are split across GPUs.
Each GPU processes only a subset of attention heads.
Results are synchronized using NCCL all_reduce.
3. Distributed Execution
Uses torchrun to launch one process per GPU.
Each process is assigned a rank and communicates via NCCL backend.
4. Cross-GPU Communication
Uses:
all_gather for MLP output reconstruction
all_reduce for attention synchronization
5. Performance Benchmarking
Measures forward-pass throughput (steps/sec)
Evaluates scaling behavior across different numbers of GPUs
Key Concepts Demonstrated
Tensor Parallelism (intra-layer model sharding)
Multi-GPU synchronization with NCCL
Transformer architecture decomposition
Distributed execution with torchrun
GPU scaling and throughput measurement
