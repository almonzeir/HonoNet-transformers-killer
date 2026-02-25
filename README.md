<div align="center">
  
  <img src="https://via.placeholder.com/800x200/0a0a0a/ffffff?text=H+O+L+O+N+E+T++v+3" alt="HoloNet Banner" width="100%">

  <h1>🌌 HoloNet v3</h1>
  <p><b>Hybrid Sliding-Window Attention × Gated Rotational Recurrent Vaults</b></p>

  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-0a0a0a.svg?style=for-the-badge"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Experimental_Prototype-00ffcc.svg?style=for-the-badge"></a>

  <br><br>

  > *"Stop tuning the toy and start architecting the machine; the world doesn't reward elegance, it rewards scale."*

</div>

<hr>

## ⚡ The Context Bottleneck is Dead

Standard Transformers are bound by $O(N^2)$ memory scaling. They must hold the entire Key-Value (KV) cache in VRAM, crippling inference at long contexts. **HoloNet v3** shatters this limitation by physically separating working memory from long-term semantic tracking.

### 1. The Sniper (Local Working Memory)
A precision Multi-Head Attention block strictly bounded to a sliding local window. It provides perfect, high-fidelity recall of the immediate context with zero long-term memory overhead.

### 2. The Vault (Global Recurrent Memory)
As tokens fall out of the sliding window, they are not discarded. They are compressed into a fixed-dimensional recurrent state ($h_t$) via a Cayley-stabilized rotational matrix and a low-rank capacity bottleneck.

<br>

## 🧬 The Core Mathematics

HoloNet utilizes a unitary rotation-based state update to prevent the vanishing gradients of traditional RNNs. The transition is governed by the **Cayley Transform**, ensuring strict orthogonality, combined with a learned decay factor ($\gamma$) and a selective Low-Rank forgetting matrix ($L$):

<div align="center">
  
  `h_t = γ (D h_{t-1}) + g_t ⊙ (L h_{t-1}) + W_{in} x_t`
  
</div>

* **$D = (I - S)(I + S)^{-1}$** : A perfectly stable, non-drifting rotation matrix.
* **$L = A B^T$** : The low-rank bottleneck enforcing semantic compression.
* **$\gamma$** : The learnable exponential decay factor filtering high-frequency noise.

<br>

## 🛑 Systems-Level Optimizations

### 1. The "Kill Switch" (Vault-Forcing Dropout)
Neural networks are lazy; they will optimize for the short-path (Local Attention) and ignore the Vault. HoloNet fixes this. During training, the local attention is stochastically masked to `0` at random intervals. **The Sniper goes blind.** The optimizer is mathematically forced to route gradients through the Vault to minimize loss, guaranteeing utilization.

### 2. $O(\log N)$ Associative Scan Readiness
The gating mechanism ($g_t$) is strictly input-dependent ($x_t$), decoupling it from the recurrent state ($h_t$). This renders the update rule a Linear Time-Invariant (LTI) system, fully compatible with parallel associative prefix-sums for near-Mamba training speeds on GPU clusters.

### 3. Dual-Regime Learning Rates
We separate the Riemannian optimization of the orthogonal matrix (slower, highly controlled) from the Euclidean optimization of the standard weights (fast), preventing catastrophic phase drift during training.

<br>

## 💻 Initialization (PyTorch)

```python
import torch
from model.holonet import HoloNetBlock

# Initialize the Hybrid Layer
# d_model: Hidden dimension size | vault_dropout: The "Kill Switch" probability
hybrid_layer = HoloNetBlock(d_model=512, n_heads=8, vault_dropout=0.2).cuda()

# Input sequence: (Batch_Size, Sequence_Length, D_Model)
x = torch.randn(4, 1024, 512).cuda()

# Forward pass: Local Attention + Global Vault
output = hybrid_layer(x)
print(output.shape) # torch.Size([4, 1024, 512])
