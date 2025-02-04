import pytorch_lightning as pl
from timm.models.densenet import re
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from scipy.stats import spearmanr
import ast
from trainer import retrieve_encoder, retrieve_decoder, AutoEncoder
from tqdm import tqdm
import yaml

def process_gsm8kcot(text):
    lines = ast.literal_eval(text)
    # merge latex fraction expressions
    merged = []
    # skip "sure lets break this down step by step"
    i=1
    while i < len(lines):
        if "\\[" in lines[i]:
            merged.append("".join(lines[i:i+3]))
            i += 3
        else:
            merged.append(lines[i])
            i += 1
    return merged

def load_and_split_data():
    dataset = load_dataset("Kanan275/GSM8k-CoT")
    train_data = dataset['train']

    # Split into train and validation
    train_idx, val_idx = train_test_split(range(len(train_data)), test_size=0.1, random_state=42)
    return train_data.select(train_idx), train_data.select(val_idx)

@torch.no_grad()
def embed_text(encoder, tokenizer, text):
    if isinstance(text, str):
        text = [text]
    with torch.no_grad():
        tokens = tokenizer(
            text,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        token_ids = torch.tensor(tokens['input_ids']).to(encoder.device)
        mask = torch.tensor(tokens['attention_mask']).to(encoder.device)

        emb = encoder.encode(token_ids, mask)
    return emb

def calculate_basic_metrics(encoder, tokenizer, validation_data, batch_size=8):
    all_metrics = []

    for item in tqdm(validation_data):
        steps = process_gsm8kcot(item['step_list'])
        if len(steps) < 2:  # Skip examples with too few steps
            continue

        # Get embeddings for all steps
        embeddings = []
        for step in steps:
            with torch.no_grad():
                emb = embed_text(encoder, tokenizer, step)
                embeddings.append(emb)
        embeddings = np.stack(embeddings).squeeze(1)
        # Calculate various metrics
        metrics = {}
        # 1. Step-wise Cosine Similarity
        cos_sims = cosine_similarity(embeddings)
        # Average similarity between consecutive steps
        consecutive_sims = np.diagonal(cos_sims, offset=1)
        metrics['avg_consecutive_similarity'] = np.mean(consecutive_sims)

        # 2. Step Order Correlation
        # Calculate similarity between each step and the final step
        final_step_sims = cos_sims[-1]
        # Expected: similarity should increase as we get closer to the final step
        expected_trend = np.arange(len(steps))
        correlation, _ = spearmanr(final_step_sims, expected_trend)
        metrics['step_order_correlation'] = correlation

        # 3. Local Coherence
        # Average similarity between each step and its immediate neighbors
        local_coherence = []
        for i in range(1, len(steps)-1):
            local_sim = (cos_sims[i][i-1] + cos_sims[i][i+1]) / 2
            local_coherence.append(local_sim)
        metrics['local_coherence'] = np.mean(local_coherence) if local_coherence else 0

        all_metrics.append(metrics)

    # Average metrics across all examples
    final_metrics = {}
    for key in all_metrics[0].keys():
        final_metrics[key] = np.mean([m[key] for m in all_metrics])

    return final_metrics

def evaluate_numerical_sensitivity(encoder, tokenizer, validation_data, num_samples=100):
    import re
    from sklearn.metrics.pairwise import cosine_similarity  # Change to sklearn's version

    def modify_numbers(text):
        def replace_number(match):
            num = int(match.group())
            if np.random.random() < 0.5:
                return str(num // 10)
            else:
                return str(num * 10)
        return re.sub(r'\d+', replace_number, text)

    sensitivity_metrics = {
        'numerical_sensitivity': [],
        'numerical_relative_change': []
    }

    for item in tqdm(validation_data):
        steps = process_gsm8kcot(item['step_list'])

        for step in steps:
            if not re.search(r'\d+', step):
                continue

            # Get original embedding
            with torch.no_grad():
                original_emb = embed_text(encoder, tokenizer, step)

            # Create modified version with changed numbers
            modified_step = modify_numbers(step)

            # Get modified embedding
            with torch.no_grad():
                modified_emb = embed_text(encoder, tokenizer, modified_step)

            # Calculate cosine similarity using sklearn's version
            cos_sim = cosine_similarity(original_emb, modified_emb)[0][0]
            cos_distance = 1 - cos_sim  # Convert similarity to distance

            # Calculate relative change in embedding norm
            orig_norm = np.linalg.norm(original_emb)
            mod_norm = np.linalg.norm(modified_emb)
            relative_change = abs(orig_norm - mod_norm) / orig_norm

            sensitivity_metrics['numerical_sensitivity'].append(cos_distance)
            sensitivity_metrics['numerical_relative_change'].append(relative_change)

            if len(sensitivity_metrics['numerical_sensitivity']) >= num_samples:
                break

        if len(sensitivity_metrics['numerical_sensitivity']) >= num_samples:
            break

    return {
        'avg_numerical_sensitivity': np.mean(sensitivity_metrics['numerical_sensitivity']),
        'std_numerical_sensitivity': np.std(sensitivity_metrics['numerical_sensitivity']),
        'avg_relative_change': np.mean(sensitivity_metrics['numerical_relative_change']),
        'std_relative_change': np.std(sensitivity_metrics['numerical_relative_change'])
    }

class GSM8kDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        steps = process_gsm8kcot(item['step_list'])
        return steps


class EvaluationModule(pl.LightningModule):
    def __init__(self, encoder, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.all_metrics = []
        self.test_results = {}
    @torch.no_grad()
    def forward(self, text):
        return embed_text(self.encoder, self.tokenizer, text)
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        steps = batch
        # Get embeddings for all steps
        embeddings = []
        for step in steps:
            emb = self(step)
            embeddings.append(emb)
        embeddings = torch.stack(embeddings).squeeze(1)

        # Calculate metrics
        metrics = self._calculate_metrics(embeddings)
        self.all_metrics.append(metrics)

    def _calculate_metrics(self, embeddings):
        # Move calculations to GPU
        cos_sims = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        metrics = {}
        # Calculate consecutive similarities
        consecutive_sims = torch.diagonal(cos_sims, offset=1)
        metrics['avg_consecutive_similarity'] = consecutive_sims.mean().item()

        # Calculate step order correlation
        final_step_sims = cos_sims[-1]
        expected_trend = torch.arange(len(final_step_sims), device=self.device)
        correlation = spearmanr(final_step_sims.cpu().numpy(), expected_trend.cpu().numpy())[0]
        metrics['step_order_correlation'] = correlation

        # Calculate local coherence
        local_coherence = []
        for i in range(1, len(embeddings)-1):
            local_sim = (cos_sims[i][i-1] + cos_sims[i][i+1]) / 2
            local_coherence.append(local_sim)
        metrics['local_coherence'] = torch.tensor(local_coherence).mean().item() if local_coherence else 0

        return metrics

    def on_test_end(self):
        # Aggregate results
        final_metrics = {}
        if self.all_metrics:
            for key in self.all_metrics[0].keys():
                final_metrics[key] = np.mean([m[key] for m in self.all_metrics])

            self.test_results = final_metrics
            # # Gather results from all GPUs
            # final_metrics = self.all_gather(final_metrics)

            # # Only process on rank 0
            # if self.global_rank == 0:
            #     # Average across all processes
            #     self.test_results = {
            #         k: np.mean([d[k] for d in final_metrics]) for k in final_metrics.keys()
            #     }

def evaluate_embeddings_pl(encoder, tokenizer, validation_data, num_gpus=2):
    torch.cuda.empty_cache()

    dataset = GSM8kDataset(validation_data, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True
    )

    model = EvaluationModule(encoder, tokenizer)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        strategy='auto',
        precision=32,
    )

    trainer.test(model, dataloader)
    # torch.distributed.barrier()
    return model.test_results
    torch.cuda.empty_cache()


def load_model(checkpoint_path, encoder_info, decoder_info):
    # Initialize your model
    model = AutoEncoder(encoder_info, decoder_info)  # Add any initialization parameters needed

    # Load the state dict
    state_dict = torch.load(checkpoint_path)

    # If the state dict was saved with DataParallel, it will have 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    return model


if __name__ == "__main__":
    with open("test_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    encoder_name = config.get("encoder_name", "bert-large")
    decoder_name = config.get("decoder_name", "llama3-3b")
    checkpoint_path = config["checkpoint_path"]
    devices = config["devices"]
    max_val_samples = config.get("n_eval", 1000)

    encoder_info = retrieve_encoder(encoder_name)
    decoder_info = retrieve_decoder(decoder_name)
    encoder = load_model(checkpoint_path, encoder_info, decoder_info)
    tokenizer = encoder_info['tokenizer']
    train_data, val_data = load_and_split_data()
    val_data = val_data.select(range(min(len(val_data), max_val_samples)))  # Optional: limit validation samples
    results = evaluate_embeddings_pl(encoder, tokenizer, val_data, devices)

    print("\nBasic Evaluation Results:")
    print(f"Average Consecutive Similarity: {results['avg_consecutive_similarity']:.4f}")
    print(f"Step Order Correlation: {results['step_order_correlation']:.4f}")
    print(f"Local Coherence: {results['local_coherence']:.4f}")
    print("\nNumerical Sensitivity Results:")
    torch.cuda.empty_cache()
