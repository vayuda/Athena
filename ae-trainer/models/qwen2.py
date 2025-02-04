import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from .cross_attention import CrossAttentionDecoderLayerWrapper, CrossAttention

class Qwen2CA(nn.Module):
    def __init__(self, model, cross_attn_layers, cross_attn_heads):
        super().__init__()
        self.model = model
        self.cross_attn_layers = cross_attn_layers

        # Get reference to decoder layers
        self.layers = self.model.model.layers
        self._latent_vector = None
        # Replace all layers with conditional cross-attention layers
        for i, layer in enumerate(self.layers):
            layer_idx = i
            apply_cross_attn = layer_idx in cross_attn_layers

            if apply_cross_attn:
                self.layers[layer_idx] = CrossAttentionDecoderLayerWrapper(
                    original_layer=layer,
                    cross_attn_module=CrossAttention(self.model.config.hidden_size, cross_attn_heads),
                    apply_cross_attn=True
                ).to(torch.bfloat16)
            else:
                self.layers[layer_idx] = CrossAttentionDecoderLayerWrapper(
                    original_layer=layer,
                    cross_attn_module=None,
                    apply_cross_attn=False
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

def get_qwen2_struct(size, use_cache=False):
    if size == "1.5b":
        model_name = "Qwen/Qwen2.5-1.5B"
        cross_attn_heads = 16  # Adjust based on model architecture
        cross_attn_layers = [5, 10, 15, 20]  # Adjust based on number of layers
    else:
        raise ValueError(f"Unsupported Qwen2 model size: {size}")

    # Load base model
    print(f"Loading model: {model_name}, using cache: {use_cache}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=use_cache,
    )
    print("Model layers configuration:")
    print(f"Layer structure: {model.model.layers}")

    # Modify model architecture
    modified_model = Qwen2CA(model, cross_attn_layers, cross_attn_heads)

    return {
        "model": modified_model,
        "tokenizer": AutoTokenizer.from_pretrained(model_name),
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
