import torch
from torch import nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.query = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Compute attention scores
        # (batch_size, seq_length)
        scores = torch.matmul(x, self.query)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum of vectors
        # (batch_size, input_dim)
        output = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        
        return output