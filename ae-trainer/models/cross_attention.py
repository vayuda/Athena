import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossAttention(nn.Module):
    def __init__(self, hidden_size, cross_attn_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = cross_attn_heads

        # Linear transformations for Q, K, V in cross-attention
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.scaling = (self.hidden_size // self.num_heads) ** -0.5

    def forward(self, hidden_states, latent_vector):
        batch_size, seq_len, _ = hidden_states.shape
        # Expand latent vector to match sequence length for cross-attention
        latent_vector_expanded = latent_vector.unsqueeze(1).expand(-1, seq_len, -1)

        # Prepare Q, K, V
        query = self.query_proj(hidden_states)
        key = self.key_proj(latent_vector_expanded)
        value = self.value_proj(latent_vector_expanded)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output


class CrossAttentionDecoderLayerWrapper(nn.Module):
    def __init__(self, original_layer, cross_attn_module, apply_cross_attn=False):
        super().__init__()
        self.original_layer = original_layer
        self.cross_attn = cross_attn_module
        self.apply_cross_attn = apply_cross_attn

    def forward(self, hidden_states, latent_vector=None, use_cache=False, past_key_value=None, **kwargs):
        # Prepare kwargs for original layer
        layer_kwargs = {
            'use_cache': use_cache,
            'past_key_value': past_key_value
        }
        layer_kwargs.update(kwargs)
        # print(f"Layer kwargs: {layer_kwargs}")  # Debug print
        # Run original layer operations
        outputs = self.original_layer(
            hidden_states,
            **layer_kwargs
        )
        # print(f"Outputs type: {type(outputs)}")  # Debug print
        # print(f"Outputs structure: {outputs}")    # Debug print

        # if isinstance(outputs, tuple):
        #     print(f"Outputs length: {len(outputs)}")  # Debug print

        present_key_value = None
        if use_cache and len(outputs)>=2:
            hidden_states, present_key_value = outputs[:2]


        # Conditionally apply cross-attention
        if self.apply_cross_attn and latent_vector is not None:
            cross_attn_output = self.cross_attn(hidden_states, latent_vector)
            hidden_states = hidden_states + cross_attn_output

        if use_cache:
            return (hidden_states, present_key_value)
        return (hidden_states,)
