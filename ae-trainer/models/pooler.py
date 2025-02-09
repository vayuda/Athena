import torch
from torch import nn
import torch.nn.functional as F

class SimpleAttentionPool(nn.Module):
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

class AttentionPool(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # Linear layers to project input to query, key, and value vectors
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)

        # Scaling factor for the dot product attention
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)

        # Project input to query, key, and value vectors
        queries = self.query_proj(x)  # (batch_size, seq_length, hidden_dim)
        keys = self.key_proj(x)       # (batch_size, seq_length, hidden_dim)
        values = self.value_proj(x)   # (batch_size, seq_length, hidden_dim)

        attn_output = F.scaled_dot_product_attention(queries, keys, values)

        # Sum over the sequence length to get the final output
        # (batch_size, hidden_dim)
        return attn_output.sum(dim=1)


class MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)
