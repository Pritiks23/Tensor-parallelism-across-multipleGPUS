I built a minimal tensor-parallel transformer inference system where attention heads and MLP layers are explicitly sharded across GPUs, and NCCL collectives are used to synchronize intermediate activations, simulating how large language models are executed in distributed GPU environments.



# Tensor Parallel Transformer (PyTorch + NCCL)

## What I built

In this project, I built a **minimal tensor-parallel transformer inference system from scratch using PyTorch distributed and NCCL collectives**.

Instead of running a transformer on a single GPU, I designed the system so that **multiple GPUs collaboratively execute a single model by splitting computation across devices**.

---

## What I am doing at a high level

I take a transformer-style model and **explicitly partition its computation across GPUs**, so that no single GPU holds or executes the full model.

---

### 1. I shard attention across GPUs (tensor parallel attention)

I split attention heads across devices:

- Each GPU is responsible for only a subset of attention heads  
- Each GPU computes attention independently on its assigned heads  
- The results are synchronized across GPUs using NCCL communication primitives  

This allows the attention computation to scale horizontally across GPUs.

---

### 2. I shard the MLP (feed-forward network)

Instead of each GPU computing the full linear transformation, I split the weight matrices:

- Each GPU holds only a slice of the MLP projection weights  
- Each GPU computes a partial output independently  
- I use collective communication (e.g., `all_gather`) to reconstruct the full output tensor  

This mirrors how large-scale transformer systems distribute feed-forward computation.

---

### 3. I use NCCL collectives to synchronize computation

To make distributed computation coherent, I rely on:

- `all_gather` → to combine partial MLP outputs  
- `all_reduce` → to synchronize attention outputs across GPUs  

These operations ensure that each GPU contributes to a single unified model output.

---

### 4. I run a full forward pass across multiple GPUs

I execute a transformer block end-to-end where:

- Input tensors are replicated across GPUs  
- Each GPU computes its shard of attention and MLP  
- Intermediate results are synchronized across devices  
- The final output represents a fully distributed forward pass  

---

## Why I built this

I built this project to understand and demonstrate **how large language models scale beyond a single GPU**.

Modern transformer models exceed the memory and compute capacity of one device, so they rely on:

- tensor parallelism (splitting computation within a layer)
- distributed communication (NCCL collectives)
- synchronized execution across GPUs

This project directly implements those core ideas in a simplified but faithful way.

---

## What the point is

This is not a predictive model project.

The goal is to demonstrate that I understand:

- how transformer computation is structured internally  
- how attention and MLP layers can be partitioned across GPUs  
- how distributed systems coordinate computation using NCCL  
- how modern LLM inference scales across hardware  

In short, I built a **distributed execution system for transformer models**, not a standalone AI application.
