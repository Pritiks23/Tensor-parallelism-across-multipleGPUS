I built a minimal tensor-parallel transformer inference system where attention heads and MLP layers are explicitly sharded across GPUs, and NCCL collectives are used to synchronize intermediate activations, simulating how large language models are executed in distributed GPU environments.



# Tensor Parallel Transformer (PyTorch + NCCL)

## What I built

In this project, I built a **tensor-parallel transformer inference system from scratch using PyTorch distributed and NCCL collectives**.

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



The goal is to demonstrate that I understand:

- how transformer computation is structured internally  
- how attention and MLP layers can be partitioned across GPUs  
- how distributed systems coordinate computation using NCCL  
- how modern LLM inference scales across hardware  

In short, I built a **distributed execution system for transformer models**

# 🧠 1. Memory scaling (the biggest real benefit)

A single GPU has fixed VRAM.

---

## Example (rough real-world numbers)

Let’s say:

- GPT-style layer weight matrix ≈ 500MB–2GB (depending on size)
- Activation memory during inference adds more

---

## On 1 GPU:

- VRAM limit: 24 GB (RTX 3090)
- Model + activations: ~22–26 GB  
→ may NOT fit or will OOM

---

## On 2 GPUs (tensor parallel):

Each GPU stores ~50% of weights:

- GPU 0: ~12 GB  
- GPU 1: ~12 GB  

Total model capacity: ~2× larger model fits

---

## On 4 GPUs:

Each GPU holds ~25% of weights  
→ allows ~4× larger model capacity

---

## ✔ Numeric takeaway

Memory scales approximately linearly with number of GPUs:

\[
\text{max model size} \propto N_{\text{GPUs}}
\]

---

# ⚡ 2. Compute speed (throughput improvement)

Now the important nuance: speed is **NOT perfectly linear**.

---

## Ideal case (theoretical)

If work is perfectly split:

- 2 GPUs → 2× speed  
- 4 GPUs → 4× speed  

---

## Real case (what actually happens)

Because of communication overhead (NCCL):

### 2 GPUs:
~1.6× to 1.9× speedup

### 4 GPUs:
~2.8× to 3.5× speedup

---

## Why not perfect scaling?

Because of:

- `all_reduce` synchronization cost  
- PCIe / NVLink bandwidth limits  
- kernel launch overhead  
- imbalance in compute vs communication  

---

## ✔ Numeric takeaway

\[
\text{speedup} =
\frac{\text{compute gain}}{\text{communication overhead}}
\]

So:

| GPUs | Ideal | Realistic |
|------|------|----------|
| 1 → 2 | 2.0× | 1.6–1.9× |
| 1 → 4 | 4.0× | 2.8–3.5× |

---

# 🔥 3. Latency vs throughput tradeoff (very important)

This is subtle but interview-critical.

---

## Latency (single request)

May NOT improve much:

- 1 GPU: 120 ms  
- 2 GPUs: 80–100 ms  

### Why?

- communication overhead is “always on”

---

## Throughput (many requests)

This is where GPUs shine:

- 1 GPU → 100 tokens/sec  
- 2 GPUs → 160–190 tokens/sec  
- 4 GPUs → 280–350 tokens/sec  

