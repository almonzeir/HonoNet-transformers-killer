"""
HoloNet Block: Component Integration Wrapper
Combines Local Attention and Vault (SSM) into a unified block
"""

import torch
import torch.nn as nn
from .attention import LocalAttention
from .vault import Vault


class HoloNetBlock(nn.Module):
    """
    A complete HoloNet block combining:
    1. Local Attention (the sniper) - for precise local dependencies
    2. Vault/SSM (state management) - for long-range state tracking
    3. Feed-forward networks - for non-linear transformations
    
    This is the core building block that can be stacked to create deep models.
    """
    
    def __init__(self, dim, num_heads=8, window_size=64, ssm_state_dim=128, 
                 dropout=0.1, ff_expansion=4):
        """
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            window_size: Local attention window size
            ssm_state_dim: State space model dimension
            dropout: Dropout probability
            ff_expansion: Feed-forward expansion factor
        """
        super().__init__()
        self.dim = dim
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Local Attention Component
        self.attention = LocalAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        # Vault/SSM Component
        self.vault = Vault(
            state_dim=ssm_state_dim,
            input_dim=dim,
            output_dim=dim
        )
        
        # Feed-Forward Network
        ff_hidden = int(dim * ff_expansion)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, dim),
            nn.Dropout(dropout)
        )
        
        # Gating mechanisms for combining attention and SSM outputs
        self.attention_gate = nn.Linear(dim, dim)
        self.vault_gate = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        Process input through integrated HoloNet block.
        
        Args:
            x: Input tensor (batch_size, seq_len, dim)
        
        Returns:
            Output tensor (batch_size, seq_len, dim)
        """
        # Residual connection through attention
        attn_out = self.attention(self.norm1(x))
        attn_gated = attn_out * torch.sigmoid(self.attention_gate(attn_out))
        x = x + attn_gated
        
        # Residual connection through SSM/Vault
        ssm_out = self.vault(self.norm2(x))
        ssm_gated = ssm_out * torch.sigmoid(self.vault_gate(ssm_out))
        x = x + ssm_gated
        
        # Residual connection through feed-forward
        ff_out = self.feed_forward(self.norm3(x))
        x = x + ff_out
        
        return x


class HoloNetStack(nn.Module):
    """
    Stack multiple HoloNet blocks to create a deep model.
    """
    
    def __init__(self, dim, num_blocks=6, num_heads=8, window_size=64, 
                 ssm_state_dim=128, dropout=0.1, ff_expansion=4):
        """
        Args:
            dim: Model dimension
            num_blocks: Number of HoloNet blocks to stack
            num_heads: Number of attention heads
            window_size: Local attention window size
            ssm_state_dim: State space model dimension
            dropout: Dropout probability
            ff_expansion: Feed-forward expansion factor
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            HoloNetBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                ssm_state_dim=ssm_state_dim,
                dropout=dropout,
                ff_expansion=ff_expansion
            )
            for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        Process input through stack of HoloNet blocks.
        
        Args:
            x: Input tensor (batch_size, seq_len, dim)
        
        Returns:
            Output tensor (batch_size, seq_len, dim)
        """
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        return x
