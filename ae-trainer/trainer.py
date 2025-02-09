import yaml
from argparse import ArgumentParser
import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

from models import SimpleAttentionPool
from data_loaders import TextDataModule

class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_info,
        decoder_info,
        pooler=SimpleAttentionPool,
        **kwargs
    ):
        super().__init__()
        # Store hyperparameters
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.temp_weight = kwargs.get('temp_loss_weight', 0)
        self.rec_weight = kwargs.get('rec_loss_weight', 1)
        self.contrast_weight = kwargs.get('contrast_loss_weight', 0.25)
        self.teacher_forcing_ratio = kwargs.get('teacher_forcing_ratio', 0.5)
        self.masked_token_prob = kwargs.get('masked_token_prob', 0.15)
        self.latent_dim = kwargs.get('latent_dim', 2048)
        self.beta = kwargs.get('beta', 1.0)  # KL weight
        self.kl_annealing_steps = kwargs.get('kl_annealing_steps', 10000)
        self.save_hyperparameters()
        self.log_var = nn.Parameter(torch.zeros(1, self.latent_dim))
        # Initialize tokenizers
        self.encoder_tokenizer = encoder_info['tokenizer']
        self.decoder_tokenizer = decoder_info['tokenizer']

        # Ensure decoder tokenizer has pad token
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # Ensure decoder tokenizer has bos token
        if self.decoder_tokenizer.bos_token is None:
            self.decoder_tokenizer.bos_token = self.decoder_tokenizer.eos_token

        self.pad_token_id = self.decoder_tokenizer.pad_token_id

        # Encoder
        base = encoder_info['model']
        for param in base.parameters():
            param.requires_grad = False
        self.encoder = get_peft_model(base, encoder_info['lora_config'])

        #Bottleneck
        self.pooler = pooler(self.encoder.config.hidden_size)
        self.up = nn.Linear(
            self.encoder.config.hidden_size,
            self.latent_dim
        )

        # Decoder
        base = decoder_info['model']
        for param in base.parameters():
            param.requires_grad = False
        self.decoder = get_peft_model(base, decoder_info['lora_config'])




    def encode(self, input_ids, attention_mask):
        # pdb.set_trace()
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = self.pooler(encoder_outputs.hidden_states[-1])
        latent = self.up(pooled)
        return latent

    def decode(self, latent, input_ids, target_ids=None, attention_mask=None):
        # pdb.set_trace()
        output = self.decoder(
            input_ids = input_ids,
            latent_vector = latent,
            labels = target_ids,
            attention_mask = attention_mask
        )

        if target_ids is not None:
            return output.loss, output.logits
        else:
            return output.logits

    def greedy_decode(self, latent: torch.Tensor, max_new_tokens: int = 64):
        device = latent.device
        batch_size = latent.size(0)

        # Initialize input with bos token
        bos_token_id = self.decoder_tokenizer.bos_token_id
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

        for step in range(max_new_tokens-1):
            logits = self.decode(latent, input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

        return input_ids

    def beam_decode(self, latent, max_new_tokens=64, beam_width=5):
        # TODO: fix caching to make this work
        device = latent.device
        batch_size = latent.size(0)

        # Initialize beams with cache
        bos_token_id = self.decoder_tokenizer.bos_token_id
        beams = [{
            'sequence': torch.full((1, 1), bos_token_id, dtype=torch.long, device=device),
            'score': 0.0,
            'cache': None  # Will store KV cache
        } for _ in range(beam_width)]

        for step in range(max_new_tokens):
            all_candidates = []

            for beam in beams:
                # Use cached computation where possible
                output = self.decoder(
                    input_ids=beam['sequence'],
                    latent_vector=latent,
                    use_cache=True,
                    past_key_values=beam['cache']
                )

                logits = output.logits[:, -1, :]  # Get only last token logits
                beam['cache'] = output.past_key_values  # Update cache

                # Get top-k candidates
                probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs[0], beam_width)

                for prob, token_id in zip(topk_probs, topk_ids):
                    new_sequence = torch.cat([beam['sequence'],
                                           token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                    all_candidates.append({
                        'sequence': new_sequence,
                        'score': beam['score'] + prob.item(),
                        'cache': beam['cache']
                    })

            # Select top-k beams
            beams = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:beam_width]

            # Early stopping if all beams ended
            if all(beam['sequence'][0, -1].item() == self.decoder_tokenizer.bos_token_id
                   for beam in beams):
                break

        return beams[0]['sequence']  # Return best beam

    def forward(self, input_ids, target_ids, attention_mask):
        latent = self.encode(input_ids, attention_mask)
        return self.decode(latent, target_ids, target_ids = target_ids, attention_mask = attention_mask)

    def training_step(self, batch, batch_idx):
        (
            src_enc_ids,
            src_dec_label_ids,
            tgt_enc_ids,
            src_enc_attention_mask,
            src_dec_attention_mask,
            tgt_enc_attention_mask,
        ) = batch
        batch_size = src_enc_ids.size(0)
        seq_len = src_enc_ids.size(1)

        # Corrupt input ids with mask tokens
        masked_input_ids = src_enc_ids.clone()
        mask = torch.rand_like(src_enc_ids, dtype=torch.float) < self.masked_token_prob
        masked_input_ids[mask] = self.encoder_tokenizer.mask_token_id
        src_latents = self.encode(masked_input_ids, src_enc_attention_mask)
        src_latents2 = self.encode(src_enc_ids, src_enc_attention_mask)  # For contrastive loss

        # Calculate KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + self.log_var - src_latents.pow(2) - self.log_var.exp())
        kl_loss = kl_loss / batch_size  # Normalize by batch size

        # Teacher forcing logic
        use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
        bos_token_id = self.decoder_tokenizer.bos_token_id
        if use_teacher_forcing:
            dec_src_ids = torch.cat([
                torch.full((batch_size, 1), bos_token_id, device=src_dec_label_ids.device),
                src_dec_label_ids[:,:-1]
            ], dim=1)
        else:
            dec_src_ids = self.greedy_decode(src_latents, max_new_tokens=seq_len)

        # Reconstruction loss
        src_ar_loss, src_logits = self.decode(src_latents, dec_src_ids, target_ids=src_dec_label_ids, attention_mask=src_dec_attention_mask)
        src_rec_loss = F.cross_entropy(
            src_logits.view(-1, src_logits.size(-1)),
            src_dec_label_ids.view(-1),
            ignore_index=self.pad_token_id,
        )

        # SimCSE loss calculation
        latents1_norm = F.normalize(src_latents, p=2, dim=1)
        latents2_norm = F.normalize(src_latents2, p=2, dim=1)
        sim_matrix = torch.matmul(latents1_norm, latents2_norm.t())
        temp = 0.05
        sim_matrix = sim_matrix / temp
        labels = torch.arange(batch_size, device=self.device)
        src_contrastive_loss = (
            F.cross_entropy(sim_matrix, labels, ignore_index=self.pad_token_id) +
            F.cross_entropy(sim_matrix.t(), labels, ignore_index=self.pad_token_id)
        ) / 2

        # Temporal contrastive loss (not currently used but interesting to track)
        tgt_latents = self.encode(tgt_enc_ids, tgt_enc_attention_mask)
        tgt_latent_norm = F.normalize(tgt_latents, p=2, dim=1)
        temp_sim_matrix = torch.matmul(latents1_norm, tgt_latent_norm.t())
        temp_sim_matrix = temp_sim_matrix / temp
        temporal_loss = F.cross_entropy(temp_sim_matrix, labels, ignore_index=self.pad_token_id)

        # ELBO loss combines reconstruction loss and KL divergence
        beta = min(self.global_step / self.kl_annealing_steps, 1.0) * self.beta
        elbo_loss = src_rec_loss + 0 * beta * kl_loss

        # Total loss combining all components
        total_loss = (
            self.rec_weight * elbo_loss +
            self.temp_weight * temporal_loss +
            self.contrast_weight * src_contrastive_loss
        )

        # Log metrics
        self.log('train_loss', total_loss, sync_dist=True)
        self.log('train_rec_loss', src_rec_loss, sync_dist=True)
        self.log('train_kl_loss', kl_loss, sync_dist=True)
        self.log('train_elbo_loss', elbo_loss, sync_dist=True)
        self.log('train_temp_loss', temporal_loss, sync_dist=True)
        self.log('train_contrastive_loss', src_contrastive_loss, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        (
            src_enc_ids,
            src_dec_label_ids,
            tgt_enc_ids,
            src_enc_attention_mask,
            src_dec_attention_mask,
            tgt_enc_attention_mask,
        ) = batch
        seq_len = src_enc_ids.size(1)

        # just focus on reconstruction loss from greedy decoding
        src_latents = self.encode(src_enc_ids, src_enc_attention_mask)
        decoder_src_input_ids = self.greedy_decode(src_latents, max_new_tokens=seq_len)
        _, src_logits = self.decode(src_latents, decoder_src_input_ids, target_ids=src_dec_label_ids, attention_mask=src_dec_attention_mask)
        total_loss = F.cross_entropy(
            src_logits.view(-1, src_logits.size(-1)),
            src_dec_label_ids.view(-1),
            ignore_index=self.pad_token_id)

        # Log metrics
        self.log('val_loss', total_loss, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.05 * total_steps)
        # Optionally can create parameter groups with different learning rates
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

# To enable easier addition of new models
def retrieve_encoder(model):
    model_name = model.split("-")[0]
    model_size = model.split("-")[1]
    if model_name == "bert":
        from models import get_bert_struct
        return get_bert_struct(model_size)
    else:
        assert False, "Invalid encoder name"


def retrieve_decoder(model, **kwargs):
    model_name = model.split("-")[0]
    model_size = model.split("-")[1]

    if model_name == "llama3":
        from models import get_llama3_struct as get_model
    elif model_name == "qwen2":
        from models import get_qwen2_struct as get_model
    else:
        from models import get_gpt2_struct as get_model

    return get_model(model_size, **kwargs)

# save model after every epoch
class CModelCheckpoint(pl.Callback):
    def __init__(self, dirpath, filename):
        self.dirpath = dirpath
        self.filename = filename

    def on_validation_end(self, trainer, pl_module):
        # Save only the model's state_dict
        torch.save(pl_module.state_dict(), f"{self.dirpath}/{self.filename}.pth")

# Track memory usage
class MemoryProfileCallback(pl.Callback):
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
    # setup weights and biases to track experiments
    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(config)
    if 'seed' in config:
        pl.seed_everything(config['seed'])

    # Initialize AE model
    encoder_info = retrieve_encoder(config['encoder'])
    decoder_info = retrieve_decoder(config['decoder'], **{"latent_dim": config['hparams']["latent_dim"]})

    # loading from a checkpoint doesn't work for some reason
    if args.load_checkpoint:
        print("Loading checkpoint...")
        wandb.init(project=config['wandb-project'],
            resume=True,
            id=config['wandb-run'],
            name=f"{config['encoder']}_{config['decoder']}_{args.v}")
        ae = AutoEncoder.load_from_checkpoint(config['checkpoint'],
            encoder_info=encoder_info,
            decoder_info=decoder_info,
            **config['hparams']
        )

    else:
        print("Initializing model...")
        ae = AutoEncoder(encoder_info = encoder_info,
            decoder_info = decoder_info,
            **config['hparams']
        )

    # Initialize wandb logger for experiment tracking
    wandb_logger = WandbLogger(
        project="lcm-vae",  # Name of your W&B project
        name=f"{config['encoder']}_{config['decoder']}_{args.v}",
        log_model=False  # Optional: logs model checkpoints
    )

    # Initialize callbacks
    model_callback = CModelCheckpoint(config["model_dir"], f"{config['encoder']}_{config['decoder']}_{args.v}")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # Where to save checkpoints locally
        filename="{epoch}-{val_loss:.2f}",  # Checkpoint file naming
        monitor="val_loss",  # Metric to monitor for best model saving
        mode="min",  # 'min' for loss, 'max' for accuracy, etc.
        save_top_k=3,  # Save the top 3 models based on the monitored metric
        save_last=True,  # Save the last checkpoint
    )

    # Initialize data module
    data_module = TextDataModule(
        file_path = config['data_path'],
        encoder_tokenizer=ae.encoder_tokenizer,
        decoder_tokenizer=ae.decoder_tokenizer,
        batch_size = config['batch_size'],
        max_length = config['max_length'],
        num_workers= config['num_workers'],
        train_ratio = config['train_ratio'],
    )
    # Train model
    trainer = pl.Trainer(logger = wandb_logger,
        callbacks=[
            checkpoint_callback,
            MemoryProfileCallback(),
            model_callback,
            # stsb_eval_callback
        ],
        max_epochs = config['max_epochs'],
        devices = config['devices'],
        accumulate_grad_batches = config['accumulate_grad_batches'],
        precision = config['tr-precision'],
        gradient_clip_val = config['gradient_clip_val'],
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu")

    trainer.fit(ae, data_module)
    wandb.finish()


if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    # model arguments
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-v", type=str, default="test")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
