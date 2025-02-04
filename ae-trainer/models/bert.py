from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import LoraConfig


def get_bert_struct(size):
    if size == 'base':
        model = AutoModelForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base",
            output_hidden_states=True  # Enable hidden states
        )
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-large",
            output_hidden_states=True  # Enable hidden states
        )
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    bert_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
                "attn.Wqkv",
                "attn.Wo",
                "mlp.Wi",
                "mlp.Wp",
            ]
    )
    return {
        'model': model,
        'lora_config': bert_lora_config,
        'tokenizer': tokenizer,
        'embedding_size': model.config.hidden_size,
        'context_length': model.config.max_position_embeddings
    }
