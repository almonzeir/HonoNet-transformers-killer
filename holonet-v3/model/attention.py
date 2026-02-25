import torch
from torch import nn

class LocalSniperAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(LocalSniperAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, attn_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        return attn_output, attn_weights


def causal_mask(size):
    # Generate a causal mask
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask
