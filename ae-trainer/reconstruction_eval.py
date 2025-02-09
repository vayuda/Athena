import yaml
import torch
from torch.nn import DataParallel
from trainer import retrieve_encoder, retrieve_decoder, AutoEncoder
from datasets import load_dataset
from tqdm import tqdm
import argparse
import ast

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

def load_model(checkpoint_path, encoder_info, decoder_info, device, **kwargs):
    """
    Loads the AutoEncoder model from the checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        encoder_info (dict): Encoder configuration.
        decoder_info (dict): Decoder configuration.
        device (torch.device): Device to load the model on.

    Returns:
        AutoEncoder: Loaded model.
    """
    # Initialize the model
    model = AutoEncoder(
        encoder_info=encoder_info,
        decoder_info=decoder_info,
        **kwargs,
    )

    # Load the state dict
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # print("Checkpoint keys:", state_dict.keys())
    # If the state dict was saved with DataParallel, remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def reconstruct_sentences(
    model,
    sentences,
    device,
    batch_size=4,
    beam_width=5,
    max_new_tokens=64,
):
    reconstructed = []
    # Get the base model if using DataParallel
    base_model = model.module if isinstance(model, DataParallel) else model

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]

            # Use base_model to access tokenizer
            tokens = base_model.encoder_tokenizer(
                batch_sentences,
                max_length=max_new_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            # Use model directly for forward passes
            latent = base_model.encode(input_ids, attention_mask).to(torch.bfloat16)

            # Use model directly for beam search
            best_seqs = base_model.greedy_decode(
                latent,
                max_new_tokens=max_new_tokens,
            )

            # Use base_model to access tokenizer
            batch_reconstructed = base_model.decoder_tokenizer.batch_decode(
                best_seqs.tolist(),
                skip_special_tokens=True
            )
            reconstructed.extend(batch_reconstructed)

        return reconstructed

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Retrieve encoder and decoder configurations
    encoder_info = retrieve_encoder(config['encoder'])
    decoder_info = retrieve_decoder(config['decoder'],**config['decoder_config'])

    # Load the AutoEncoder model
    print("Loading AutoEncoder model...")
    ae_model = load_model(
        checkpoint_path=config['checkpoint_path'],
        encoder_info=encoder_info,
        decoder_info=decoder_info,
        device=device,
        **config['hparams']
    ).to(torch.bfloat16)
    if torch.cuda.device_count() > 1:
        ae_model = DataParallel(ae_model)

    print("testing model fluency: ")
    tokens = ae_model.encoder_tokenizer(
        ["The quick brown fox jumps over the lazy dog.", "\\[\\frac{1}{2} + \\frac{1}{3}\\]"],
        max_length=30,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Use model directly for forward passes
    with torch.no_grad():
        latent = ae_model.encode(input_ids, attention_mask).to(torch.bfloat16)

    # Use model directly for beam search
    with torch.no_grad():
        best_seqs = ae_model.greedy_decode(
            latent,
            max_new_tokens=30,
        )
    print(best_seqs)
    # Use base_model to access tokenizer
    batch_reconstructed = ae_model.decoder_tokenizer.batch_decode(
        best_seqs.tolist(),
        skip_special_tokens=True
    )
    print(batch_reconstructed)
    # exit()

    # Load GSM8k-CoT dataset
    print("Loading GSM8k-CoT dataset...")
    dataset = load_dataset("Kanan275/GSM8k-CoT")
    validation_data = dataset['train']

    # Select 100 examples
    max_examples = config['n_eval']
    selected_data = validation_data.select(range(max_examples))
    print(f"Selected {len(selected_data)} examples for reconstruction.")

    # Prepare output file
    output_file = args.output if args.output else "reconstructions.txt"
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write('Original |<><>| Reconstructed\n\n')
        # Iterate over each example
        for idx, example in enumerate(tqdm(selected_data, desc="Reconstructing")):
            steps = example['step_list']
            # Treat each sentence independently
            original_sentences = process_gsm8kcot(steps)
            reconstructed_sentences = reconstruct_sentences(
                model=ae_model,
                sentences=original_sentences,
                device=device,
                max_new_tokens=config['max_seq_len']
            )

            # Write to file
            for i in range(len(original_sentences)):
                f_out.write(original_sentences[i] + " |<><>| " + reconstructed_sentences[i] + "\n")
            f_out.write("\n" + "="*50 + "\n\n")

    print(f"Reconstruction completed. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct GSM8k-CoT Sentences using AutoEncoder")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--output", type=str, default="reconstructions.txt", help="Path to the output txt file")
    args = parser.parse_args()
    main(args)
