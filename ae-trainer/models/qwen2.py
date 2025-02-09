from timm.models.layers import cbam
import torch.nn as nn
import torch
from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast
from peft import LoraConfig
from .cross_attention import CrossAttentionDecoderLayerWrapper, CrossAttention

class Qwen2CA(nn.Module):
    def __init__(self, model, latent_dim):
        super().__init__()
        self.model = model
        if self.model.config.hidden_size % 64 != 0:
            raise ValueError("Hidden size must be divisible by 64")
        nheads = self.model.config.hidden_size // 64

        # Get reference to decoder layers
        self.layers = self.model.model.layers
        self._latent_vector = None
        # Replace  layers with cross-attention layers
        # half = len(self.layers) // 2
        for i, layer in enumerate(self.layers):
            cam = CrossAttention(self.model.config.hidden_size, latent_dim, nheads)
            self.layers[i] = CrossAttentionDecoderLayerWrapper(
                original_layer=layer,
                cross_attn_module=cam,
            ).to(torch.bfloat16)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            latent_vector=None,
            labels=None,
            use_cache=False,
            past_key_values=None,
        ):
            if latent_vector is not None:
                latent_vector = latent_vector
            else:
                print("no latent found")
                latent_vector = self._latent_vector

            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_vector=latent_vector,
                labels=labels,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )

def get_qwen2_struct(size, **kwargs):
    print(kwargs)
    use_cache = kwargs.get("use_cache", False)
    latent_dim = kwargs.get("latent_dim", 4096)
    if size == "1.5b":
        model_name = "Qwen/Qwen2.5-1.5B"
    elif size == "3bcode":
        model_name = "Qwen/Qwen2.5-Coder-3B"
    else:
        raise ValueError(f"Unsupported Qwen2 model size: {size}")

    # Load base model
    print(f"Loading model: {model_name}, using cache: {use_cache}")
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=use_cache,
    )

    # Modify model architecture
    modified_model = Qwen2CA(model, latent_dim)

    return {
        "model": modified_model,
        "tokenizer": Qwen2TokenizerFast.from_pretrained(model_name),
        "lora_config": LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none"
        ),
        "context_length": model.config.max_position_embeddings,
        "hidden_size": model.config.hidden_size,
    }
