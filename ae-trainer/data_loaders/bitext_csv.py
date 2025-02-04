import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

class TextDataset(Dataset):
    def __init__(self, data, encoder_tokenizer, decoder_tokenizer, max_length=128):
        """
        Args:
            data (pd.DataFrame): DataFrame containing 'source' and 'target' columns
            encoder_tokenizer: Tokenizer for source text
            decoder_tokenizer: Tokenizer for target text
            max_length (int): Maximum sequence length for tokenization
        """
        self.data = data
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source_text = row["source"]
        target_text = row["target"]

        # todo: add domain specific noise to the source text such as replacing \frac{a}{b} with \frac{mask}{b}

        # Tokenize source and target texts
        source_encoder_tokens = self.encoder_tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        source_decoder_tokens = self.decoder_tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        target_encoder_tokens = self.encoder_tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        # Input IDs
        src_enc_ids = torch.tensor(source_encoder_tokens['input_ids'], dtype=torch.long)
        src_dec_ids = torch.tensor(source_decoder_tokens['input_ids'], dtype=torch.long)
        tgt_enc_ids = torch.tensor(target_encoder_tokens['input_ids'], dtype=torch.long)

        # Attention masks
        src_enc_attention_mask = torch.tensor(source_encoder_tokens['attention_mask'], dtype=torch.long)
        src_dec_attention_mask = torch.tensor(source_decoder_tokens['attention_mask'], dtype=torch.long)
        tgt_enc_attention_mask = torch.tensor(target_encoder_tokens['attention_mask'], dtype=torch.long)

        return (
            src_enc_ids,
            src_dec_ids,
            tgt_enc_ids,
            src_enc_attention_mask,
            src_dec_attention_mask,
            tgt_enc_attention_mask,
        )

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path,
        encoder_tokenizer,
        decoder_tokenizer,
        batch_size=32,
        max_length=512,
        num_workers=4,
        train_ratio=0.8,
        random_state=42
    ):
        super().__init__()
        self.file_path = file_path
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.random_state = random_state

    def setup(self, stage=None):
        # Load the entire dataset
        with open(self.file_path,'rb') as f:
            df = pickle.load(f)

        # Split into train and validation
        train_df, val_df = train_test_split(
            df,
            train_size=self.train_ratio,
            random_state=self.random_state
        )

        # Create dataset objects
        self.train_dataset = TextDataset(
            train_df,
            self.encoder_tokenizer,
            self.decoder_tokenizer,
            self.max_length
        )

        self.val_dataset = TextDataset(
            val_df,
            self.encoder_tokenizer,
            self.decoder_tokenizer,
            self.max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )
