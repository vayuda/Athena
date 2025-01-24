import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer, GPT2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from transformers import GPT2Config
from peft import LoraConfig

# Modifies gpt2 block to include cross-attention
class GPT2CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Original self-attention
        self.self_attn = GPT2Attention(config)
        
        # Add cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True
        )
        
        # Additional layer norm for cross attention
        self.ln_cross = nn.LayerNorm(config.n_embd)
        
        # Rest of the block remains the same
        self.mlp = GPT2MLP(4,config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, hidden_states, context_vector, attention_mask=None):
        # Self attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Create causal mask and combine with attention mask
        seq_length = hidden_states.size(-2)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        causal_mask = causal_mask.to(hidden_states.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask & ~causal_mask
        else:
            attention_mask = ~causal_mask.unsqueeze(0).unsqueeze(1).expand(hidden_states.size(0), 1, seq_length, seq_length)
        attention_mask = attention_mask.float()
        # pdb.set_trace()
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        
        hidden_states = residual + hidden_states
        
        # Cross attention
        residual = hidden_states
        hidden_states = self.ln_cross(hidden_states)
        
        # Expand context vector to match sequence length
        # context_vector shape: (batch_size, 1, n_embd)
        context = context_vector.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        # Apply cross attention
        hidden_states = self.cross_attn(
            query=hidden_states,
            key=context,
            value=context
        )[0]
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

# full gpt2 decoder with cross-attention
class GPT2WithCrossAttention(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2CrossAttentionBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self, 
        input_ids, 
        context_vector, 
        position_ids=None, 
        attention_mask=None,
        labels=None,
    ):
        # Standard GPT2 embedding
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Pass through transformer blocks with cross-attention
        for block in self.h:
            hidden_states = block(hidden_states, context_vector, attention_mask)
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate next token prediction loss
            loss_fct = nn.CrossEntropyLoss()
            next_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = next_token_loss

        return (loss, logits) if loss is not None else logits


def get_gpt_config(size):
    if size == 'small':
        cfg = {
                'n_embd': 768,        # Embedding dimension
                'n_head': 12,         # Number of attention heads
                'n_layer': 12,        # Number of transformer blocks
                'n_positions': 1024,  # Max sequence length
                'vocab_size': 50257,  # GPT-2's vocabulary size
                'attn_pdrop': 0.1,    # Attention dropout probability
                'embd_pdrop': 0.1,    # Embedding dropout probability
                'resid_pdrop': 0.1,   # Residual dropout probability
                'activation_function': 'gelu'  # Activation function
        }
    elif size == 'medium':
        cfg = {
                'n_embd': 1024,        # Embedding dimension
                'n_head': 16,         # Number of attention heads
                'n_layer': 24,        # Number of transformer blocks
                'n_positions': 1024,  # Max sequence length
                'vocab_size': 50257,  # GPT-2's vocabulary size
                'attn_pdrop': 0.1,    # Attention dropout probability
                'embd_pdrop': 0.1,    # Embedding dropout probability
                'resid_pdrop': 0.1,   # Residual dropout probability
                'activation_function': 'gelu'  # Activation function
        }
    elif size == 'large':
        cfg = {
                'n_embd': 1280,        # Embedding dimension
                'n_head': 20,         # Number of attention heads
                'n_layer': 36,        # Number of transformer blocks
                'n_positions': 1024,  # Max sequence length
                'vocab_size': 50257,  # GPT-2's vocabulary size
                'attn_pdrop': 0.1,    # Attention dropout probability
                'embd_pdrop': 0.1,    # Embedding dropout probability
                'resid_pdrop': 0.1,   # Residual dropout probability
                'activation_function': 'gelu'  # Activation function
        }
    return GPT2Config(**cfg)


def get_gpt2_struct(size='small'):
    gpt_config = get_gpt_config(size)
    model = GPT2WithCrossAttention(gpt_config)
    gpt2_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn", "c_proj"]  # GPT2 attention layer names
    )
    return {
        'model': model,
        'config': gpt_config,
        'lora_config': gpt2_lora_config,
        'tokenizer': GPT2Tokenizer.from_pretrained('gpt2'),
        'embedding_size': gpt_config.n_embd,
        'vocab_size': gpt_config.vocab_size,
        'context_length': gpt_config.n_positions
    }