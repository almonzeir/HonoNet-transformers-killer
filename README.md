# 🌌 HoloNet v3: Hybrid Sliding-Window Attention with Gated Rotational Recurrent Vaults

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

> **"Stop tuning the toy and start architecting the machine; the world doesn't reward elegance, it rewards scale."**

HoloNet v3 is an experimental hybrid neural architecture designed to solve the $O(N^2)$ context bottleneck of traditional Transformers. It fuses the exact retrieval capabilities of **Sliding-Window Attention** with the infinite-horizon compression of a **Cayley-Stabilized Rotational Vault**.

## 🧠 The Architecture

Standard Transformers must keep the entire Key-Value (KV) cache in memory, leading to massive hardware bottlenecks during long-context inference. HoloNet fixes this by splitting the memory into two distinct modules:

1. **The Sniper (Local Working Memory):** A standard Multi-Head Attention block restricted to a sliding local window (e.g., 2048 tokens) for perfect, high-fidelity recall of recent context.
2. **The Vault (Global Recurrent Memory):** As tokens fall out of the sliding window, they are compressed into a fixed-dimensional recurrent state ($h_t$). 

### The Vault Mathematics
To prevent the vanishing/exploding gradients that plague traditional RNNs, HoloNet utilizes a unitary rotation-based state update, mathematically constrained by the **Cayley Transform** to ensure strict orthogonality, alongside a learned decay factor ($\gamma$) and a Low-Rank matrix ($L$) for selective forgetting:

$$h_t = \gamma (D h_{t-1}) + g_t \odot (L h_{t-1}) + W_{in} x_t$$

Where:
* $D = (I - S)(I + S)^{-1}$ is a perfectly stable rotation matrix.
* $L = A B^T$ is the low-rank capacity bottleneck forcing semantic compression.
* $\gamma$ is a learnable exponential decay factor to clear high-frequency noise.

## 🚀 Key Innovations

* **The "Kill Switch" (Vault-Forcing Attention Dropout):** To solve the "Lazy Optimization" problem where gradient descent ignores the recurrent state, HoloNet implements stochastic windowing. During training, the local attention is dynamically masked to `0` at random intervals, mathematically forcing the optimizer to route gradients through the Vault to minimize loss.
* **Associative Scan Readiness:** The gating mechanism ($g_t$) is strictly input-dependent ($x_t$), decoupling it from the recurrent state ($h_t$). This ensures the update rule is a Linear Time-Invariant (LTI) system, making it compatible with $O(\log N)$ parallel associative scans (prefix sums) for Mamba-like training speeds on modern GPUs.
* **Dual-Regime Learning Rates:** HoloNet separates the Riemannian optimization of the orthogonal matrix from the Euclidean optimization of the standard weights, preventing catastrophic phase drift during training.

## 💻 Quick Start (PyTorch)

```python
import torch
from holonet import HoloNetBlock

# Initialize the Hybrid Layer
# d_model: Hidden dimension size
# vault_dropout: The "Kill Switch" probability during training
hybrid_layer = HoloNetBlock(d_model=512, n_heads=8, vault_dropout=0.2).cuda()

# Input sequence: (Batch_Size, Sequence_Length, D_Model)
x = torch.randn(4, 1024, 512).cuda()

# Forward pass combining Local Attention and the Global Vault
output = hybrid_layer(x)
print(output.shape) # torch.Size([4, 1024, 512])
