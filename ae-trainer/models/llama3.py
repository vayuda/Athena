import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from .cross_attention import CrossAttentionDecoderLayerWrapper, CrossAttention


class Llama3CA(nn.Module):
    def __init__(self, model, cross_attn_layers, cross_attn_heads):
        super().__init__()
        self.model = model
        self.cross_attn_layers = cross_attn_layers

        # Get reference to decoder layers
        self.layers = self.model.model.layers

        # Replace all layers with conditional cross-attention layers
        for i, layer in enumerate(self.layers):
            layer_idx = i  # Assuming layers are indexed from 0
            apply_cross_attn = layer_idx in cross_attn_layers

            if apply_cross_attn is not None:
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
            context_vector=None,
            labels=None,
            use_cache=False,
            past_key_values=None,
        ):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_vector=context_vector,
                labels=labels,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )

def get_llama3_struct(size, use_cache=False):
    if size == "3b":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cross_attn_heads = 24
        cross_attn_layers = [5, 10, 15, 20, 25]
    else:
        model_name = "meta-llama/Llama-3.1-8B"
        cross_attn_heads = 32
        cross_attn_layers = [5, 10, 15, 20, 25, 30]

    # Load base model
    print(f"Loading model: {model_name}, using cache: {use_cache}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=use_cache,
    )

    # Modify model architecture
    modified_model = Llama3CA(model, cross_attn_layers, cross_attn_heads)

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
