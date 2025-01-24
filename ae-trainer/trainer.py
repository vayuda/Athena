import yaml
from argparse import ArgumentParser
import pdb
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback



import wandb
from peft import LoraConfig, get_peft_model

from models import AttentionPool


# full AE model with BERT encoder, GPT2 decoder, and attention pooler
class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_info,
        decoder_info,
        pooler=AttentionPool,
        learning_rate=1e-4,
        ar_loss_weight= 0.1,
        noise_scale = 0.1,
        context_length = 512,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize tokenizers
        self.encoder_tokenizer = encoder_info['tokenizer']
        self.decoder_tokenizer = decoder_info['tokenizer']
        
        # Ensure decoder tokenizer has pad token
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        
        # Initialize encoder
        base = encoder_info['model']
        for param in base.parameters():
            param.requires_grad = False
        self.encoder = get_peft_model(base, encoder_info['lora_config'])
        
        # AE bottleneck
        self.pooler = pooler(self.encoder.config.hidden_size)
        
        # Project latent space to decoder hidden dimension
        self.latent_proj = nn.Linear(
            self.encoder.config.hidden_size, 
            decoder_info['embedding_size']  # Use config value
        )
        
        # Initialize decoder
        base = decoder_info['model']
        for param in base.parameters():
            param.requires_grad = False
        self.decoder = get_peft_model(base, decoder_info['lora_config'])
        
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.ar_loss_weight = ar_loss_weight
        self.noise_scale = noise_scale
        self.context_length = context_length

    def encode(self, input_ids, attention_mask):
        # pdb.set_trace()
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = self.pooler(encoder_outputs.last_hidden_state)
        latent = self.latent_proj(pooled)
        return latent
    
    def decode(self, latent, input_ids, target_ids=None, attention_mask=None):
        # pdb.set_trace()
        # add noise while training
        if self.training:
            noise = torch.randn_like(latent) * self.noise_scale
            latent += noise
            return self.decoder(input_ids, latent, labels = target_ids, attention_mask = attention_mask)
        else:
            if target_ids is not None:
                return self.decoder(input_ids, latent, labels = target_ids, attention_mask = attention_mask)
            else:
                return self.decoder(input_ids, latent, attention_mask = attention_mask)

    def forward(self, input_ids, target_ids, attention_mask):
        latent = self.encode(input_ids, attention_mask)
        return self.decode(latent, target_ids, target_ids = target_ids, attention_mask = attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, target_ids, attention_mask = batch
        
        # Forward pass
        latent = self.encode(input_ids, attention_mask)
        ar_loss, logits = self.decode(latent, target_ids, attention_mask)
        # Calculate cross entropy reconstruction loss
        rec_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # KL divergence loss
        #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = rec_loss + self.ar_loss_weight * ar_loss #+ kl_loss
        
        # Log metrics
        self.log('train_loss', total_loss)
        self.log('train_rec_loss', rec_loss)
        self.log('train_ar_loss', ar_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        encoder_input_ids, decoder_input_ids, attention_mask = batch
        
        # Forward pass
        ar_loss, logits = self(encoder_input_ids, decoder_input_ids, attention_mask)
        
        # Calculate loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), decoder_input_ids.view(-1)) + self.ar_loss_weight * ar_loss
        
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


# DataModule for handling the bitext CSV data
class TextAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        val_csv_path,
        encoder_tokenizer,
        decoder_tokenizer,
        batch_size=32,
        max_length=512,
        num_workers=4,
        source_col='source',
        target_col='target',
        delimiter='\t'
    ):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.source_col = source_col
        self.target_col = target_col
        self.delimiter = delimiter
        self.num_workers = num_workers
        
    def prepare_data(self):
        # CSV data will be read during setup
        pass
        
    def setup(self, stage=None):
        from datasets import load_dataset
        # Load datasets without streaming
        train_dataset = load_dataset('csv', 
            data_files=self.train_csv_path,
            delimiter=self.delimiter)
        
        val_dataset = load_dataset('csv', 
            data_files=self.val_csv_path,
            delimiter=self.delimiter)
        
        self.train_dataset = TextDataset(
            dataset=train_dataset,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            max_length=self.max_length
        )
        
        self.val_dataset = TextDataset(
            dataset=val_dataset,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            max_length=self.max_length
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Validation data shouldn't be shuffled
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
            prefetch_factor=2
        )

    
# Dataset class for bitext data
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, encoder_tokenizer, decoder_tokenizer, max_length, source_col='source', target_col='target'):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_length = max_length
        self.source_col = source_col
        self.target_col = target_col
        
        # Load the entire dataset into memory
        self.dataset = list(dataset['train'])
        
    def __len__(self):
        return len(self.dataset)
            
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        source_text = item[self.source_col]
        target_text = item[self.target_col]
        
        encoder_tokens = self.encoder_tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        decoder_tokens = self.decoder_tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        return (
            torch.tensor(encoder_tokens['input_ids']),
            torch.tensor(decoder_tokens['input_ids']),
            torch.tensor(encoder_tokens['attention_mask'])
        )


def retrieve_encoder(model):
    model_name = model.split("-")[0]
    model_size = model.split("-")[1]
    if model_name == "bert":
        from models import get_bert_struct
        return get_bert_struct(model_size)
    else:
        assert False, "Invalid encoder name"


def retrieve_decoder(model):
    model_name = model.split("-")[0]
    model_size = model.split("-")[1]
    if model_name == "gpt2":
        from models import get_gpt2_struct
        return get_gpt2_struct(model_size)
    else:
        assert False, "Invalid decoder name"


class MemoryProfileCallback(Callback):
    def __init__(self):
        self.has_printed = False
        
    def on_after_backward(self, trainer, pl_module):
        if not self.has_printed:
            stats = torch.cuda.memory_stats()
            print("\n=== Memory Stats After First Backward ===")
            print(f"Allocated memory: {stats['allocated_bytes.all.current'] / 1024**2:.2f}MB")
            print(f"Peak allocated: {stats['allocated_bytes.all.peak'] / 1024**2:.2f}MB")
            print(f"Reserved memory: {stats['reserved_bytes.all.current'] / 1024**2:.2f}MB")
            self.has_printed = True

def main(args):
    torch.set_float32_matmul_precision('medium')
    # setup weights and biases to track experiments
    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    

    #initialize encoder and decoder
    encoder_info = retrieve_encoder(config['encoder'])
    decoder_info = retrieve_decoder(config['decoder'])
    context_length = min(encoder_info['context_length'], decoder_info['context_length'])
    # Initialize AE model
    if args.load_checkpoint:
        print("Loading checkpoint...")
        wandb.init(project=config['wandb-project'],
            resume=True,
            id=config['wandb-run'],
            name=f"{config['encoder']}_{config['decoder']}_{args.v}")
        ae = AutoEncoder.load_from_checkpoint(config['checkpoint'],
            encoder_info=encoder_info,
            decoder_info=decoder_info)

    else:
        print("Initializing model...")
        ae = AutoEncoder(encoder_info = encoder_info,
            decoder_info = decoder_info,
            learning_rate = config['learning_rate'],
            ar_loss_weight= config['ar_loss_weight'],
            noise_scale = config['noise_scale'],
        )
    print('model initialized, showing vram usage:')
    # print(torch.cuda.memory_stats())
    total_params = 0
    params_with_grad = 0

    # for name, param in ae.named_parameters():
    #     total_params += 1
    #     if param.requires_grad:
    #         params_with_grad += 1
    #         print(f"Parameter requiring grad: {name}")

    # print(f"\nParameters with gradients: {params_with_grad}/{total_params}")
    # exit()
    wandb_logger = WandbLogger(
        project="lcm-vae",  # Name of your W&B project
        name=f"{config['encoder']}_{config['decoder']}_{args.v}",      
        log_model="all"              # Optional: logs model checkpoints
    )
    # Configure checkpointing callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # Where to save checkpoints locally
        filename="{epoch}-{val_loss:.2f}",  # Checkpoint file naming
        monitor="val_loss",  # Metric to monitor for best model saving
        mode="min",  # 'min' for loss, 'max' for accuracy, etc.
        save_top_k=3,  # Save the top 3 models based on the monitored metric
        save_last=True,  # Save the last checkpoint 
    )

    # print("Model parameters:")
    # for name, module in ae.named_modules():
    #     print(f"{"  "}{name}")


    # Initialize data module
    data_module = TextAEDataModule(
        train_csv_path = config['train_data_path'],
        val_csv_path = config['validation_data_path'],
        encoder_tokenizer=ae.encoder_tokenizer,
        decoder_tokenizer=ae.decoder_tokenizer,
        batch_size = config['batch_size'],
        max_length = context_length,
    )
    
    # Train model
    trainer = pl.Trainer(logger = wandb_logger,
        callbacks=[checkpoint_callback, MemoryProfileCallback()],     
        max_epochs = config['max_epochs'],
        devices = config['devices'],
        accumulate_grad_batches = config['accumulate_grad_batches'],
        precision = config['tr-precision'],
        gradient_clip_val = config['gradient_clip_val'],
        strategy = config['mgpu-strategy'],
        accelerator="gpu")

    trainer.fit(ae, data_module)
    wandb.finish()


    

    # Test reconstruction with a sample text
    # sample_text = "This is a test sentence to check reconstruction."
    # reconstruct_text(ae, sample_text)

if __name__ == "__main__":
    parser = ArgumentParser()
    # model arguments
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-v", type=str, default="test")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    