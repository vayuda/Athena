import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr, pearsonr
import numpy as np

class STSBDataset(Dataset):
    def __init__(self, split='validation', tokenizer=None, max_length=128):
        self.dataset = load_dataset('sentence-transformers/stsb', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['sentence1'], item['sentence2'], item['score']

class STSBEDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32, max_length=128, num_workers=2, split='validation'):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.split = split

    def setup(self, stage=None):
        self.stsb_dataset = STSBDataset(split=self.split, tokenizer=self.tokenizer, max_length=self.max_length)

    def train_dataloader(self):
        # Optionally, define if needed
        pass

    def val_dataloader(self):
        # Optionally, define if needed
        pass

    def test_dataloader(self):
        return DataLoader(
            self.stsb_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

class STSBEvaluator(pl.LightningModule):
    def __init__(self, model, batch_size=16):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

        # Create lists to store predictions and targets
        self.all_predictions = []
        self.all_targets = []

    def forward(self, sentence1, sentence2):
        # Get embeddings for both sentences
        emb1 = self.model.encode(sentence1['input_ids'], sentence1['attention_mask'])
        emb2 = self.model.encode(sentence2['input_ids'], sentence2['attention_mask'])

        # Normalize embeddings
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Compute cosine similarity and scale to 0-1
        similarities = torch.sum(emb1 * emb2, dim=1)
        similarities = (similarities + 1) / 2
        return similarities

    def test_step(self, batch, batch_idx):
        sentences1, sentences2, scores = batch

        # Tokenize both sentences
        encoded1 = self.model.encoder_tokenizer(
            sentences1,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        encoded2 = self.model.encoder_tokenizer(
            sentences2,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Get predictions
        predictions = self(encoded1, encoded2)

        # Store predictions and targets
        self.all_predictions.append(predictions)
        self.all_targets.append(scores)

        # Calculate metrics for this batch
        mse = torch.mean((predictions - scores) ** 2)
        self.log('test_mse', mse)
        return mse

    def on_test_epoch_end(self):
        # Concatenate all predictions and targets
        predictions = torch.cat(self.all_predictions).cpu().numpy()
        targets = torch.cat(self.all_targets).cpu().numpy()

        # Calculate final metrics
        spearman_corr, _ = spearmanr(predictions, targets)
        pearson_corr, _ = pearsonr(predictions, targets)
        mse = np.mean((predictions - targets) ** 2)

        # Log metrics
        self.log('test_spearman', spearman_corr * 100)
        self.log('test_pearson', pearson_corr * 100)
        self.log('test_mse', mse)

        # Print results
        print(f"\nFinal Results:")
        print(f"Spearman correlation: {spearman_corr * 100:.2f}%")
        print(f"Pearson correlation: {pearson_corr * 100:.2f}%")
        print(f"Mean Squared Error: {mse:.4f}")

        # Clear stored predictions and targets
        self.all_predictions = []
        self.all_targets = []

    def test_dataloader(self):
        dataset = STSBDataset(
            split='validation',  # or 'test' for final evaluation
            tokenizer=self.model.encoder_tokenizer,
            max_length=self.model.encoder_tokenizer.model_max_length
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False
        )


class STSBEvaluationCallback(pl.Callback):
    def __init__(self, model, tokenizer, batch_size, max_length, num_workers):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        stsb_data_module = STSBEDataModule(
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            num_workers=self.num_workers,
            split='validation'
        )
        stsb_data_module.setup()
        stsb_evaluator = STSBEvaluator(model=self.model, batch_size=self.batch_size)
        with torch.no_grad():
            trainer.test(stsb_evaluator, datamodule=stsb_data_module)
        pl_module.train()
