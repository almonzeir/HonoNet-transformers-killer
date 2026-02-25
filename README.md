# 🌌 HoloNet v3: Hybrid Sliding-Window Attention × Gated Rotational Vaults

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch)
![License: MIT](https://img.shields.io/badge/License-MIT-0a0a0a.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Experimental_Prototype-00ffcc.svg?style=for-the-badge)

> *"Stop tuning the toy and start architecting the machine; the world doesn't reward elegance, it rewards scale."*

### Table of Contents
- [The Context Bottleneck](#-the-context-bottleneck)
- [Architecture Overview](#-architecture-overview)
- [Mathematical Foundation](#-mathematical-foundation)
- [Systems-Level Optimizations](#-systems-level-optimizations)
- [Quick Start](#-quick-start)
- [Roadmap](#-roadmap)

---

## ⚡ The Context Bottleneck
Standard Transformers are fundamentally crippled by $O(N^2)$ memory scaling. To maintain context, they must hold the entire Key-Value (KV) cache in VRAM, leading to catastrophic inference bottlenecks at 32k+ tokens. 

**HoloNet v3** resolves this by physically decoupling working memory from long-term semantic tracking, achieving linear $O(N)$ scaling without sacrificing local precision.

## 🧬 Architecture Overview
HoloNet replaces the standard Transformer block with a dual-pathway memory engine:

1. **The Sniper (Local Working Memory):** A precision Multi-Head Attention block restricted strictly to a sliding window (e.g., 2048 tokens). It guarantees perfect, high-fidelity recall of the immediate context with zero long-term memory overhead.
   
2. **The Vault (Global Recurrent Memory):** As tokens fall out of the sliding window, they are not discarded. They are systematically compressed into a fixed-dimensional recurrent state vector ($h_t$) using a rotation-based State Space Model (SSM) architecture.

# Mathematical Foundation
To prevent the vanishing and exploding gradients that plague traditional recurrent networks, HoloNet utilizes a unitary rotation-based state update. This is mathematically constrained by the **Cayley Transform** to ensure strict orthogonality:

$$
h_t = \gamma (D h_{t-1}) + g_t \odot (L h_{t-1}) + W_{in} x_t
$$

* **The Rotation Matrix ($D$):** Computed as $D = (I - S)(I + S)^{-1}$. This guarantees a perfectly stable rotation matrix that preserves phase coherence over infinite horizons.
* **The Information Bottleneck ($L$):** A low-rank projection matrix defined as $L = A B^T$, forcing the network to compress semantic meaning.
* **The Forgetting Mechanism ($\gamma$):** A learnable exponential decay factor designed to gently filter out high-frequency noise.

##  Systems-Level Optimizations

Reviewer critiques of hybrid architectures often point to "Lazy Optimization" and sequential hardware bottlenecks. HoloNet v3 engineers around these fatal flaws:

* **The "Kill Switch" (Vault-Forcing Dropout):** Neural networks are inherently lazy; they will over-optimize for the Local Attention path and ignore the Vault. HoloNet stochastically blinds the local attention to `0` during training, mathematically forcing the optimizer to route gradients through the Vault to minimize loss.
* **Associative Scan Readiness:** The gating mechanism ($g_t$) is strictly input-dependent ($x_t$). This decoupling renders the update rule a Linear Time-Invariant (LTI) system, fully compatible with $O(\log N)$ parallel associative prefix-sums for near-Mamba training speeds on modern GPUs.
* **Dual-Regime Learning Rates:** HoloNet isolates the Riemannian optimization of the orthogonal matrix from the Euclidean optimization of the standard weights, preventing catastrophic phase drift.








# Quick Start

```python
import torch
from model.holonet import HoloNetBlock

# 1. Initialize the Hybrid Layer 
# d_model: Hidden dimension size | vault_dropout: The "Kill Switch" probability
hybrid_layer = HoloNetBlock(d_model=512, n_heads=8, vault_dropout=0.2).cuda()

# 2. Input sequence: (Batch_Size, Sequence_Length, D_Model)
x = torch.randn(4, 1024, 512).cuda()

# 3. Forward pass combining Local Attention and the Global Vault
output = hybrid_layer(x)
print(f"Residual Stream Output: {output.shape}") 
# Expected: torch.Size([4, 1024, 512])
