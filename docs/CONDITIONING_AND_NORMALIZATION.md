# Conditioning and Normalization: Why RMSNorm + FiLM?

This document explains the different conditioning and normalization options explored in this project and why RMSNorm combined with FiLM conditioning emerged as the best choice for our transformer-based world model.

## Table of Contents

- [Normalization Options](#normalization-options)
  - [LayerNorm](#layernorm)
  - [RMSNorm](#rmsnorm)
  - [Why RMSNorm Wins](#why-rmsnorm-wins)
- [Conditioning Mechanisms](#conditioning-mechanisms)
  - [Additive Conditioning](#additive-conditioning)
  - [Concatenation](#concatenation)
  - [Cross-Attention](#cross-attention)
  - [Adaptive Layer Norm (AdaLN)](#adaptive-layer-norm-adaln)
  - [FiLM (Feature-wise Linear Modulation)](#film-feature-wise-linear-modulation)
  - [Why FiLM Wins](#why-film-wins)
- [The Winning Combination: RMSNorm + FiLM](#the-winning-combination-rmsnorm--film)
- [Implementation Details](#implementation-details)
- [Experimental Evidence](#experimental-evidence)

---

## Normalization Options

### LayerNorm

Layer Normalization normalizes across the feature dimension by computing mean and variance:

```python
# LayerNorm: normalize using both mean and variance
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
x_norm = (x - mean) / sqrt(var + eps)
return x_norm * weight + bias
```

**Pros:**
- Well-established in transformers (original "Attention is All You Need")
- Centers and scales the distribution

**Cons:**
- Two learnable parameters per feature (weight and bias)
- Mean subtraction can destroy information in the signal magnitude
- Slightly more compute than RMSNorm

### RMSNorm

Root Mean Square Normalization normalizes only by the RMS (root mean square), without centering:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS: sqrt(mean(x^2))
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight
```

**Pros:**
- Simpler: only one learnable parameter (no bias)
- Preserves relative magnitude information (no mean subtraction)
- ~10-15% faster than LayerNorm
- Better gradient flow in deep networks

**Cons:**
- Less theoretical grounding than LayerNorm

### Why RMSNorm Wins

1. **Training Stability**: In our experiments, switching from LayerNorm to RMSNorm improved training stability, especially in deeper networks.

2. **Computational Efficiency**: RMSNorm is faster because it skips mean computation and uses only one learnable parameter.

3. **Modern Adoption**: RMSNorm is used in state-of-the-art models including LLaMA, Gemma, and many diffusion transformers.

4. **Compatibility with FiLM**: RMSNorm pairs naturally with FiLM conditioning. Since RMSNorm doesn't center the features (no bias term), FiLM's shift parameter has a cleaner role. With LayerNorm, you'd have both the LN bias and FiLM's shift competing.

---

## Conditioning Mechanisms

When building conditional generative models, we need to inject conditioning information (time, action, context) into the network. Here are the main approaches:

### Additive Conditioning

The simplest approach: add the conditioning embedding to the hidden states.

```python
# Additive: just add the embedding
x = x + time_embed(t)  # After each layer
```

**Pros:**
- Simple to implement
- Minimal parameters

**Cons:**
- Limited expressivity (can only shift features)
- Cannot modulate feature scales
- Struggles with complex conditioning

### Concatenation

Concatenate the conditioning with input tokens.

```python
# Concatenation: add conditioning as extra tokens
cond_tokens = cond_embed(c).unsqueeze(1)  # [B, 1, D]
x = torch.cat([cond_tokens, x], dim=1)     # [B, N+1, D]
x = transformer(x)
x = x[:, 1:, :]  # Remove conditioning token
```

**Pros:**
- Conditioning participates in attention
- Good for variable-length conditioning

**Cons:**
- Increases sequence length
- Indirect modulation through attention
- No explicit scale control

### Cross-Attention

Use the conditioning as keys/values in cross-attention.

```python
# Cross-attention: condition attends to context
Q = self.q(x)              # Query from input
K = self.k_cond(context)   # Key from conditioning
V = self.v_cond(context)   # Value from conditioning
attn = softmax(Q @ K.T) @ V
```

**Pros:**
- Powerful for complex/variable conditioning (e.g., text-to-image)
- Each position can attend to relevant conditioning

**Cons:**
- Expensive: adds full attention computation
- Overkill for simple conditioning (time, discrete actions)
- More parameters

### Adaptive Layer Norm (AdaLN)

Used in the original DiT paper. Predict LayerNorm parameters from conditioning.

```python
# AdaLN: predict LN scale and shift from conditioning
scale, shift = self.adaLN_modulation(cond).chunk(2, dim=-1)
x = layernorm(x)
x = x * (1 + scale) + shift
```

**Pros:**
- Direct modulation of normalization
- Used in DiT paper with good results

**Cons:**
- Tied to LayerNorm specifically
- Predicts both scale and shift from single projection

### FiLM (Feature-wise Linear Modulation)

Learn separate scale and shift transformations from the conditioning signal.

```python
class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)  # Scale predictor
        self.beta = nn.Linear(dim, dim)   # Shift predictor

    def forward(self, x, cond):
        scale = 1 + self.gamma(cond)  # Centered at 1
        shift = self.beta(cond)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)
```

**Pros:**
- Separate learnable projections for scale and shift (more expressive)
- Scale initialized around 1 (identity-like at init)
- Works with any normalization
- Computationally cheap (just two linear layers)

**Cons:**
- More parameters than additive (but minimal)

### Why FiLM Wins

1. **Expressivity**: Unlike additive conditioning, FiLM can both scale and shift features. This is crucial for generative models where the conditioning should control *how much* of a feature to express, not just shift its value.

2. **Efficiency**: Unlike cross-attention, FiLM adds minimal computation—just two linear projections per block.

3. **Decoupled Projections**: FiLM uses separate `gamma` and `beta` networks, giving independent capacity for learning scales vs shifts. AdaLN chunks a single projection.

4. **Stable Initialization**: By using `1 + gamma(cond)` for scale, the network starts close to identity. This aids training stability.

5. **Flexibility**: FiLM is normalization-agnostic. We apply it *after* RMSNorm, cleanly separating normalization from modulation.

---

## The Winning Combination: RMSNorm + FiLM

The combination works so well because of clean separation of concerns:

```
Input → RMSNorm (normalize scale) → FiLM (modulate based on condition) → Attention/MLP
```

**Why this ordering matters:**

1. **RMSNorm** ensures features have unit RMS, preventing activation explosion/vanishing
2. **FiLM** then modulates these normalized features based on the conditioning signal
3. The conditioning has a consistent, normalized "canvas" to work with

**Comparison to alternatives:**

| Approach | Scale Control | Shift Control | Compute | Parameters |
|----------|---------------|---------------|---------|------------|
| Additive | ✗ | ✓ | Low | Low |
| Cross-Attention | ✓ (indirect) | ✓ (indirect) | High | High |
| AdaLN | ✓ | ✓ | Medium | Medium |
| **RMSNorm + FiLM** | ✓ | ✓ | Low | Low |

---

## Implementation Details

Our TransformerBlock applies RMSNorm + FiLM before both attention and MLP:

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        # Pre-attention
        self.norm1 = RMSNorm(dim)
        self.film1 = FiLM(dim)

        # Attention
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # Pre-MLP
        self.norm2 = RMSNorm(dim)
        self.film2 = FiLM(dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        # Attention block with conditioning
        x = x + self.attn(self.film1(self.norm1(x), cond))
        # MLP block with conditioning
        x = x + self.mlp(self.film2(self.norm2(x), cond))
        return x
```

**Key design choices:**

- **Pre-norm architecture**: Norm before attention/MLP (more stable than post-norm)
- **FiLM after norm**: Modulate normalized features, not raw activations
- **Residual connections**: Skip connections around the modulated blocks
- **Two FiLM modules per block**: Attention and MLP can be modulated differently

---

## Experimental Evidence

From our experiments on the world model task:

| Model | Normalization | Conditioning | KID ↓ |
|-------|---------------|--------------|-------|
| world_base | LayerNorm | Additive | 0.157 |
| world_filmcond | RMSNorm | FiLM | 0.128 |
| world_filmcond_pre | RMSNorm | FiLM (pretrained) | 0.106 |
| world_flow_clsfg | RMSNorm | FiLM + CFG | **0.102** |

**Key observations:**

1. **FiLM vs Additive**: Switching from additive to FiLM conditioning reduced KID from 0.157 to 0.128 (~18% improvement)

2. **Pretraining helps**: Starting from an unconditional diffusion model and finetuning with conditioning achieved better results (0.106 vs 0.128)

3. **CFG synergy**: Classifier-free guidance works well with FiLM, achieving our best result of 0.102 KID

4. **Training stability**: The RMSNorm + FiLM combination showed smoother loss curves compared to LayerNorm + additive

---

## Summary

**RMSNorm** is preferred over LayerNorm because:
- Faster (no mean computation)
- Simpler (one parameter instead of two)
- Better gradient flow
- No information loss from mean subtraction

**FiLM** is preferred over other conditioning mechanisms because:
- Expressive (controls both scale and shift)
- Efficient (just two linear layers)
- Stable initialization (scale centered at 1)
- Normalization-agnostic design

The combination provides the best of both worlds: stable training dynamics from RMSNorm and expressive conditioning from FiLM, with minimal computational overhead.

---

## References

- [RMSNorm: Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) - Uses AdaLN
- [LLaMA](https://arxiv.org/abs/2302.13971) - Uses RMSNorm
