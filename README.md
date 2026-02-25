# 🌌 HoloNet v3
**Hybrid Sliding-Window Attention × Gated Rotational Recurrent Vaults**

> *"Stop tuning the toy and start architecting the machine; the world doesn't reward elegance, it rewards scale."*

## ⚡ The Context Bottleneck is Dead
Standard Transformers are crippled by $O(N^2)$ memory scaling. They must hold the entire Key-Value cache in VRAM. **HoloNet v3** shatters this limitation by splitting memory processing into two distinct engines:

1. **The Sniper (Local Working Memory):** A precision Multi-Head Attention block restricted to a sliding window (e.g., 2048 tokens) for perfect recall of immediate context.
2. **The Vault (Global Recurrent Memory):** As tokens fall out of the sliding window, they are compressed into a fixed-dimensional recurrent state ($h_t$) via a Cayley-stabilized rotational matrix and a Low-Rank bottleneck.

## 🧬 The Core Mathematics
HoloNet utilizes a unitary rotation-based state update to prevent vanishing gradients, mathematically constrained by the **Cayley Transform** to ensure strict orthogonality:

`h_t = γ (D h_{t-1}) + g_t ⊙ (L h_{t-1}) + W_{in} x_t`

* **$D$**: A perfectly stable rotation matrix $(I - S)(I + S)^{-1}$.
* **$L$**: The low-rank bottleneck enforcing semantic compression ($A B^T$).
* **$\gamma$**: A learnable exponential decay factor filtering high-frequency noise.

## 🚀 Systems-Level Optimizations
* **The "Kill Switch" (Vault-Forcing Dropout):** Neural networks are lazy and will ignore recurrent states if given the chance. HoloNet stochastically blinds the local attention to `0` during training, mathematically forcing the optimizer to route gradients through the Vault.
* **Associative Scan Readiness:** The gating mechanism ($g_t$) is strictly input-dependent, rendering the update rule a Linear Time-Invariant (LTI) system compatible with $O(\log N)$ parallel associative prefix-sums.
* **Dual-Regime Learning Rates:** Separates the Riemannian optimization of the orthogonal matrix from the Euclidean optimization of the standard weights, preventing catastrophic phase drift.

## 💻 Quick Start (PyTorch)

```python
import torch
from model.holonet import HoloNetBlock

# Initialize the Hybrid Layer (d_model=512, vault_dropout=20%)
hybrid_layer = HoloNetBlock(d_model=512, n_heads=8, vault_dropout=0.2).cuda()

# Input sequence: (Batch_Size, Sequence_Length, D_Model)
x = torch.randn(4, 1024, 512).cuda()

# Forward pass: Local Attention + Global Vault
output = hybrid_layer(x)
print(output.shape) # torch.Size([4, 1024, 512])
