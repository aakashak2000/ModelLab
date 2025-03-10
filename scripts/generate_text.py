# scripts/generate_text.py

import os
import argparse
import json
import torch
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.basic_rnn.basic_rnn import CharRNN

def load_model(model_path, mappings_path, embedding_size, hidden_size, num_layers, dropout, device):
    """Load a trained CharRNN model."""
    # Load character mappings
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    
    vocab_size = mappings['vocab_size']
    char_to_idx = {k: int(v) for k, v in mappings['char_to_idx'].items()}
    idx_to_char = {int(k): v for k, v in mappings['idx_to_char'].items()}
    
    # Initialize model
    model = CharRNN(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Load model state
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, char_to_idx, idx_to_char

def generate_samples(model, char_to_idx, idx_to_char, seeds, max_length=200, temperature=1.0):
    """Generate text samples starting from the provided seed texts."""
    samples = []
    
    for seed in seeds:
        generated_text = model.generate_text(
            initial_text=seed,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            max_length=max_length,
            temperature=temperature
        )
        samples.append(generated_text)
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained CharRNN model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--mappings_path', type=str, required=True, help='Path to the character mappings file')
    parser.add_argument('--output_file', type=str, default='results/generated_samples/samples.txt', help='Path to save generated samples')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum length of each generated sample')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (higher = more diverse)')
    parser.add_argument('--seed_text', type=str, default=None, help='Seed text for generation (if not provided, random seeds will be used)')
    parser.add_argument('--embedding_size', type=int, default=128, help='Size of character embeddings')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability (0 for inference)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.model_path}...')
    model, char_to_idx, idx_to_char = load_model(
        model_path=args.model_path,
        mappings_path=args.mappings_path,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device
    )
    
    # Prepare seed texts
    if args.seed_text:
        seeds = [args.seed_text]
    else:
        # Use some common starting phrases
        seeds = [
            "The ",
            "Once upon a time",
            "In the beginning",
            "It was a dark",
            "She looked at",
        ]
    
    # Generate samples
    print(f'Generating {len(seeds)} samples with temperature {args.temperature}...')
    model.eval()
    samples = generate_samples(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seeds=seeds,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    # Save samples
    with open(args.output_file, 'w') as f:
        for i, sample in enumerate(samples):
            f.write(f"Sample {i+1}:\n")
            f.write(f"{sample}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f'Generated samples saved to {args.output_file}')
    
    # Also print samples to console
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(sample)
        print("-" * 50)

if __name__ == '__main__':
    main()
