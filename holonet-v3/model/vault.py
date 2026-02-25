import torch
import torch.nn as nn

class HoloNetVault(nn.Module):
    def __init__(self, d_model, rank=16):
        super().__init__()
        self.d_model = d_model
        
        # 1. Cayley Transform Parameter
        self.S_params = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        
        # 2. Learnable Decay Factor (\gamma)
        self.gamma = nn.Parameter(torch.ones(1) * 2.0)
        
        # 3. The Low-Rank Bottleneck (L = A * B^T)
        self.A = nn.Parameter(torch.randn(d_model, rank) / (d_model ** 0.5))
        self.B = nn.Parameter(torch.randn(rank, d_model) / (rank ** 0.5))
        
        # 4. Input-Dependent Gating (Strictly Associative)
        self.gate_proj = nn.Linear(d_model, d_model)
        
        # 5. Information Injection
        self.W_in = nn.Linear(d_model, d_model)

    def get_rotation_matrix(self):
        """ Computes stable orthogonal matrix D """
        S = (self.S_params - self.S_params.T) / 2
        I = torch.eye(self.d_model, device=S.device)
        D = torch.linalg.solve(I + S, I - S)
        return D

    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device
        
        h_t = torch.zeros(batch_size, self.d_model, device=device)
        D = self.get_rotation_matrix()
        gamma_decay = torch.sigmoid(self.gamma)
        
        output_states = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            g_t = torch.sigmoid(self.gate_proj(x_t))
            low_rank_update = (h_t @ self.B.T) @ self.A.T
            h_t = gamma_decay * (h_t @ D.T) + (g_t * low_rank_update) + self.W_in(x_t)
            output_states.append(h_t)
            
        return torch.stack(output_states, dim=1)