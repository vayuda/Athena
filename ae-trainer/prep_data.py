from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import re
import argparse
import torch
from typing import List
import ast
import pickle

def nll(prob):
    epsilon = 1e-10
    prob = torch.clamp(prob, epsilon, 1 - epsilon)
    return -torch.log(prob)


def find_patch_boundaries(
    next_token_nll: torch.Tensor,
    relative_threshold: float = 0.1,
    min_patch_size: int = 1,
    max_patch_size: int = 100
) -> List[int]:
    """
    Find patch boundaries using approximate monotonicity constraint on entropy.

    Args:
        token_probs: List of probability distributions for each token
        relative_threshold: Minimum relative increase in entropy to create boundary
        min_patch_size: Minimum tokens between boundaries
        max_patch_size: Maximum tokens between boundaries

    Returns:
        List of indices where patch boundaries should be placed
    """
    # Calculate surprise (negative log prob) for each position

    boundaries = [0]  # Always start with boundary at beginning
    current_pos = min_patch_size  # Start looking after minimum patch size

    while current_pos < len(next_token_nll):
        # Force boundary if we've reached maximum patch size
        if current_pos - boundaries[-1] >= max_patch_size:
            boundaries.append(current_pos)
            current_pos += min_patch_size
            continue

        # Check if entropy increases significantly compared to previous position
        prev_surprise = next_token_nll[current_pos - 1]
        curr_surprise = next_token_nll[current_pos]

        relative_increase = (curr_surprise - prev_surprise) / (prev_surprise + 1e-10)

        if relative_increase > relative_threshold:
            # We found a boundary point
            boundaries.append(current_pos)
            current_pos += min_patch_size
        else:
            current_pos += 1

    if boundaries[-1] != len(next_token_nll) - 1:
        boundaries.append(len(next_token_nll) - 1)

    return boundaries

def bert_entropy_split_into_thoughts(text, tokenizer, model, edge_threshold=0.3, kernel_size=5):
    text = text.replace('\n', ' ')
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits[0], dim=-1)
    actual_token_ids = inputs.input_ids[0][1:]  # Skip CLS token
    next_token_probs = probs[:-1].gather(1, actual_token_ids.unsqueeze(-1)).squeeze()
    next_token_nll = nll(next_token_probs)

    # Find split points using convolution
    split_points = find_patch_boundaries(next_token_nll,relative_threshold = 0.3, min_patch_size = 10, max_patch_size = 20)

    # Split text into chunks based on these points
    thoughts = []
    start_idx = 0

    for split_idx in split_points:
        chunk_tokens = inputs.input_ids[0][start_idx:split_idx+1]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if len(chunk_text.strip()) > 0:
            thoughts.append(chunk_text.strip())
        start_idx = split_idx + 1

    # Add final chunk
    if start_idx < len(inputs.input_ids[0]):
        last_chunk = tokenizer.decode(inputs.input_ids[0][start_idx:],
                                    skip_special_tokens=True)
        if len(last_chunk.strip()) > 0:
            thoughts.append(last_chunk.strip())

    return thoughts[:-1], thoughts[1:]

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

    # create bigrams
    return merged[:-1], merged[1:]


def get_sentences(text, method='commas', tokenizer=None, model=None):
    if method == 're':
        lines = re.findall(r'.*?(?:[1-9]?\.|<\/?\w*>)', text)
        return lines[:-1], lines[1:]
    elif method == 'bert' and tokenizer and model:
        return bert_entropy_split_into_thoughts(text, tokenizer, model)
    elif method == 'commas':
        return process_gsm8kcot(text)


def main(args):
    # Load dataset
    dataset = load_dataset(args.dataset)

    # Initialize tokenizer and model if needed
    if args.split_method == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-large")
    else:
        tokenizer = None
        model = None

    # Initialize lists to store sentence pairs
    targets = []
    source = []
    # Process each answer
    for answer in tqdm(dataset["train"][args.tc]):
        s, t = get_sentences(answer, method=args.split_method, tokenizer=tokenizer, model=model)
        targets.extend(t)
        source.extend(s)
        if len(targets) > args.dataset_size:
            break

    # Create DataFrame with answers as both source and target
    df = pd.DataFrame({
        'source': source,
        'target': targets
    })
    if len(df) > args.dataset_size:
        df = df.iloc[:args.dataset_size]

    # Save DataFrames to TSV files
    dataset_name = args.dataset.split('/')[-1]
    with open(f'{args.output_dir}/{dataset_name}_{args.dataset_size}.pickle',"wb") as f:
        pickle.dump(df, f)


parser = argparse.ArgumentParser(description='Convert huggingface cot datasets to sentence pairs where the target is the next sentence')
parser.add_argument('--dataset', type=str, help='huggingface dataset path')
parser.add_argument('--tc', type=str, help='Column name for the text training data')
parser.add_argument('--dataset_size', type=int, default=10000, help='Number of samples to use')
parser.add_argument('--output_dir', type=str, default='data/',
                    help='Output directory for train and validation files')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--split_method', type=str, default='commas', choices=['re', 'bert','commas'])

args = parser.parse_args()
main(args)
