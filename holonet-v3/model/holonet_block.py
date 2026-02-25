import torch
import torch.nn as nn
from .vault import HoloNetVault
from .attention import LocalSniperAttention

class HoloNetBlock(nn.Module):
    def __init__(self, d_model, n_heads, vault_rank=16, vault_dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.vault_dropout = vault_dropout
        
        # 1. The Fast Working Memory (Sniper)
        self.sniper = LocalSniperAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        
        # 2. The Persistent Global Memory (Vault)
        self.vault = HoloNetVault(d_model=d_model, rank=vault_rank)
        self.ln_vault = nn.LayerNorm(d_model)
        
        # 3. Final Processing
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # --- A. Local Attention ---
        attn_out = self.sniper(x, x, x)
        
        # 🚨 THE KILL SWITCH 🚨
        if self.training and torch.rand(1).item() < self.vault_dropout:
            attn_out = torch.zeros_like(attn_out)
            
        x = self.ln1(x + attn_out)
        
        # --- B. Global Vault ---
        vault_out = self.vault(x)
        x = x + self.ln_vault(vault_out)
        
        # --- C. FFN Processing ---
        ffn_out = self.ffn(x)
        out = self.ln2(x + ffn_out)
        
        return out
