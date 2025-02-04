import torch
import torch.nn as nn
import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .util import worker_init_fn

class PAWSDataset(Dataset):
    def __init__(self, split='train', tokenizer=None, max_length=128, num_workers=4, worker_id=0):
        # Configure sharding
        self.num_shards = num_workers
        self.shard_id = worker_id

        # Load dataset with sharding configuration
        self.dataset = load_dataset(
            'paws',
            'labeled_final',
            split=f'{split}[{self.shard_id}%{self.num_shards}]',  # Sharding syntax
            streaming=True
        ).shuffle(seed=42)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.iterator = iter(self.dataset)

        # Adjust length based on sharding
        total_length = {"train": 49401, "validation": 8000, "test": 8000}[split]
        self.length = total_length // self.num_shards + (1 if total_length % self.num_shards > self.shard_id else 0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            item = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataset)
            item = next(self.iterator)

        return item['sentence1'], item['sentence2'], item['label']





class PAWSFineTuner(L.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-5,
        similarity_threshold=0.5,
        batch_size=16,
        weight_decay=0.01,
        log_training=False
    ):
        super().__init__()
        self.model = model.encoder
        # Explicitly freeze the autoencoder
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.log_training = False
        # Add classification head
        latent_dim = self.model.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),  # 2x latent dim because we concatenate two sentences
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # For storing validation predictions
        self.all_predictions = []
        self.all_scores = []
        self.all_targets = []

    def forward(self, sentence1, sentence2):
        # Get embeddings for both sentences
        emb1 = self.model.encode(sentence1['input_ids'], sentence1['attention_mask'])
        emb2 = self.model.encode(sentence2['input_ids'], sentence2['attention_mask'])

        # Concatenate the embeddings
        combined = torch.cat([emb1, emb2], dim=1)

        # Pass through classifier
        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        sentences1, sentences2, labels = batch

        # Tokenize sentences
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

        # Forward pass
        logits = self(encoded1, encoded2)

        # Calculate loss
        loss = nn.BCELoss()(logits.squeeze(), labels.float())

        # Log metrics
        if self.log_training:
            self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        sentences1, sentences2, labels = batch

        # Tokenize sentences
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
        scores = self(encoded1, encoded2)
        predictions = (scores > self.similarity_threshold).float()

        # Store for epoch end metrics
        self.all_predictions.append(predictions)
        self.all_scores.append(scores)
        self.all_targets.append(labels)

        # Calculate loss
        loss = nn.BCELoss()(scores.squeeze(), labels.float())
        self.log('val_loss', loss)
        return loss

    def on_test_epoch_end(self):
        # Concatenate all predictions and targets
        predictions = torch.cat(self.all_predictions).cpu().numpy()
        scores = torch.cat(self.all_scores).cpu().numpy()
        targets = torch.cat(self.all_targets).cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')
        auc_roc = roc_auc_score(targets, scores)

        # Log metrics
        self.log('val_accuracy', accuracy)
        self.log('val_f1', f1)
        self.log('val_auc_roc', auc_roc)

        # Clear stored predictions
        self.all_predictions = []
        self.all_scores = []
        self.all_targets = []

    def configure_optimizers(self):
        # Only optimize classifier parameters
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    # In PAWSFineTuner, modify your dataloader methods:
    def train_dataloader(self):
        dataset = PAWSDataset(
            split='train',
            tokenizer=self.model.encoder_tokenizer
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self):
        dataset = PAWSDataset(
            split='validation',
            tokenizer=self.model.encoder_tokenizer
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            worker_init_fn=worker_init_fn
        )
