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
