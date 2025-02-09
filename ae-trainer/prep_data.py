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

# currently only supports single column text data but designed to be extended to support bitext
def regex_sentence_splitter(text: str) -> List[str]:
    # Remove any newlines and extra spaces
    text = re.sub(r'\s+', ' ', text.strip())

    # Define sentence ending patterns
    # This pattern looks for:
    # - Period followed by space and capital letter
    # - Question mark or exclamation mark followed by space
    # - Numbered lists (e.g., "1.", "2.")
    # - Common abbreviations like "Mr.", "Dr.", "St." are handled as exceptions
    pattern = r'(?<![A-Z][a-z]\.)(?<!\s[A-Z]\.)(?<=\.|!|\?)\s+(?=[A-Z])|\d+\.\s+'

    # Split the text based on the pattern
    sentences = re.split(pattern, text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

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

def get_sentences(text, method, tokenizer=None, model=None):
    if method == 're':
        lines = regex_sentence_splitter(text)
        return lines[:-1], lines[1:]
    elif method == 'pylist':
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
    for answer in tqdm(dataset[args.split][args.tc]):
        s, t = get_sentences(answer, args.split_method, tokenizer=tokenizer, model=model)
        targets.extend(t)
        source.extend(s)
        if args.dataset_size and len(targets) > args.dataset_size:
            break

    # Create DataFrame with answers as both source and target
    df = pd.DataFrame({
        'source': source,
        'target': targets
    })
    if args.dataset_size and len(df) > args.dataset_size:
        df = df.iloc[:args.dataset_size]

    # Save DataFrames to TSV files
    dataset_name = args.dataset.split('/')[-1]
    with open(f'{args.output_dir}/{dataset_name}_{args.dataset_size}.pickle',"wb") as f:
        pickle.dump(df, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert huggingface cot datasets to sentence pairs where the target is the next sentence')
    parser.add_argument('--dataset', type=str, help='huggingface dataset path')
    parser.add_argument('--split', type=str, default='train', choices=['train','validation','test'],
                        help='Split of the dataset to use')
    parser.add_argument('--tc', type=str, help='Column name for the text training data')
    parser.add_argument('--dataset_size', type=int, default=None, help='Number of samples to use')
    parser.add_argument('--output_dir', type=str, default='data/',
                        help='Output directory for train and validation files')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--split_method', type=str, default='re', choices=['re','pylist'])

    args = parser.parse_args()
    main(args)
