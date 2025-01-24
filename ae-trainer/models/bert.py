import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from peft import LoraConfig




def get_bert_struct(size):
    if size == 'base':
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif size == 'large':
        model = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    bert_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value"]  # BERT attention layer names
    )
    return {
        'model': model,
        'config': model.config,
        'lora_config': bert_lora_config,
        'tokenizer': tokenizer,
        'embedding_size': model.config.hidden_size,
        'context_length': model.config.max_position_embeddings
    }