Token IDs
   ↓
Embedding (replicated)
   ↓
TP Attention (sharded heads)
   ↓
TP MLP (sharded projection)
   ↓
Logits (gathered)
   ↓
Sampling loop


implemented a minimal Megatron-style tensor-parallel transformer block from scratch using PyTorch distributed and NCCL collectives, where attention heads and MLP projections are explicitly sharded across GPUs.”
