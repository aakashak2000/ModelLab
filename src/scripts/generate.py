"""
Generate text using a trained model.

This script loads a trained model and vocabulary, then generates text samples.

Usage:
    python generate_text.py --model basic_rnn --model-path src/models/basic_rnn_best.pt

Author: ModelLab
"""

import argparse
import os
import sys
import importlib
import torch
from pathlib import Path

# Add the parent directory to the path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


def load_vocab(vocab_path):
    """Load vocabulary information.
    
    Args:
        vocab_path (str): Path to saved vocabulary file
        
    Returns:
        dict: Vocabulary information
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    
    vocab_info = torch.load(vocab_path)
    return vocab_info


def decode_text(indices, idx_to_char, char_level=True):
    """Convert indices back to text.
    
    Args:
        indices (list): List of token indices
        idx_to_char (dict): Mapping from indices to tokens
        char_level (bool): If True, join without spaces
        
    Returns:
        str: Decoded text
    """
    
    tokens = [idx_to_char[idx if isinstance(idx_to_char, dict) else idx] for idx in indices]
    if char_level:
        return ''.join(tokens)
    else:
        return ' '.join(tokens)


def generate_text(model, seed_text, char_to_idx, idx_to_char, max_length=500, 
                 temperature=1.0, char_level=True, device='cpu'):
    """Generate text using a trained model.
    
    Args:
        model: The language model
        seed_text (str): Starting text for generation
        char_to_idx (dict): Mapping from tokens to indices
        idx_to_char (dict): Mapping from indices to tokens
        max_length (int): Maximum length to generate
        temperature (float): Sampling temperature
        char_level (bool): If True, tokenize by character
        device (str): Device to run on
        
    Returns:
        str: Generated text
    """
    model.to(device)
    model.eval()
    
    # Tokenize the seed text
    if char_level:
        tokens = list(seed_text)
    else:
        tokens = seed_text.split()
    
    # Convert to indices, handling unknown tokens
    seed_indices = []
    for token in tokens:
        # Use a default value (usually 0) for unknown tokens
        
        if token in char_to_idx:
            seed_indices.append(char_to_idx[token])
        else:
            print(f"Warning: Token '{token}' not in vocabulary, skipping.")
    
    if not seed_indices:
        raise ValueError("No valid tokens in seed text")
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            seed_indices, 
            max_length, 
            temperature=temperature, 
            device=device
        )
    
    # Decode the generated sequence
    generated_text = decode_text(generated, idx_to_char, char_level)
    
    return generated_text


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Generate text using a trained model')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture to use (e.g., basic_rnn)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--vocab-path', type=str, default=None,
                        help='Path to vocabulary file (defaults to model_name_vocab.pt)')
    
    # Generation arguments
    parser.add_argument('--seed-text', type=str, default=None,
                        help='Starting text for generation (if not provided, will use a random one)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--max-length', type=int, default=500,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower=more conservative, higher=more creative)')
    
    # Other arguments
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--output-file', type=str, default=None,
                        help='File to save generated text (defaults to console output)')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Original data file (only needed if no vocab file and no seed text)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return
    
    # Set vocab path if not provided
    if not args.vocab_path:
        args.vocab_path = args.model_path.replace('_best.pt', '_vocab.pt').replace('_final.pt', '_vocab.pt')
    
    # Load vocabulary
    try:
        vocab_info = load_vocab(args.vocab_path)
        char_to_idx = vocab_info['char_to_idx']
        idx_to_char = vocab_info['idx_to_char']
        vocab_size = vocab_info['vocab_size']
        char_level = vocab_info.get('char_level', True)
        print(f"Loaded vocabulary with {vocab_size} tokens")
    except FileNotFoundError:
        if not args.data_file:
            print(f"Error: Vocabulary file not found at {args.vocab_path} and no data file provided.")
            print("Please provide either a vocabulary file or the original training data file.")
            return
        
        print(f"Vocabulary file not found. Creating from data file {args.data_file}...")
        # Quick tokenization to recreate vocabulary
        with open(args.data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        char_level = True  # Default to character-level if not specified
        tokens = list(text) if char_level else text.split()
        chars = sorted(list(set(tokens)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        print(f"Created vocabulary with {vocab_size} tokens")
    
    # Try to dynamically import the model
    try:
        # Determine the correct module path and filename
        if args.model == 'basic_rnn':
            model_path = "src.architectures.basic_rnn.model"
            model_file = "model"
            model_class_name = "BasicRNN"
        else:
            model_path = f"src.architectures.{args.model}.{args.model}"
            model_file = args.model
            
            # Handle different naming patterns based on model type
            if args.model == 'rnn':
                model_class_name = "RNN"
            elif args.model == 'lstm':
                model_class_name = "LSTM"
            elif args.model == 'gru':
                model_class_name = "GRU"
            elif args.model == 'seq2seq':
                model_class_name = "Seq2Seq"
            elif args.model == 'attention':
                model_class_name = "Attention"
            elif args.model == 'transformer':
                model_class_name = "Transformer"
            elif args.model == 'miniGPT':
                model_class_name = "miniGPT"
            else:
                # Fallback for any other models
                model_class_name = ''.join(word.capitalize() for word in args.model.split('_'))
        
        # Import the module
        module = importlib.import_module(model_path)
        
        # Get the model class
        model_class = getattr(module, model_class_name)
        
        print(f"Successfully imported {model_class_name} from {model_path}")
    except (ImportError, AttributeError) as e:
        print(f"Error importing model: {e}")
        print(f"Make sure the model exists at src/architectures/{args.model}/model.py")
        print(f"and contains a class named {model_class_name}")
        return
    
    # Load the trained model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=torch.device(args.device))
    
    # Get model configuration
    model_config = checkpoint.get('model_config', {})
    
    # Create model with saved parameters if available
    model = model_class(
        vocab_size=model_config.get('vocab_size', vocab_size),
        embedding_dim=model_config.get('embedding_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 2),
        dropout=0.0  # No dropout needed for generation
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # Generate samples
    all_samples = []
    print(f"\nGenerating {args.num_samples} samples with temperature {args.temperature}:")
    
    for i in range(args.num_samples):
        # Use provided seed text or create a simple default
        if i == 0 and args.seed_text:
            seed_text = args.seed_text
        else:
            # For subsequent samples or if no seed provided, use a simple default
            # In a real application, you might extract random seeds from your training data
            seed_text = "The " if char_level else "The"
        
        print(f"\nSample {i+1}:")
        print(f"Seed: '{seed_text}'")
        
        # Generate text
        try:
            generated_text = generate_text(
                model=model,
                seed_text=seed_text,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                max_length=args.max_length,
                temperature=args.temperature,
                char_level=char_level,
                device=args.device
            )
            
            # Print result
            print(f"Generated: '{generated_text}'")
            print("-" * 50)
            
            # Store the sample
            all_samples.append({'seed': seed_text, 'generated': generated_text})
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
    
    # Save to output file if specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(all_samples):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Seed: {sample['seed']}\n")
                f.write(f"Generated: {sample['generated']}\n")
                f.write("-" * 50 + "\n\n")
        print(f"Samples saved to {args.output_file}")
    
    print("Done!")


if __name__ == '__main__':
    main()