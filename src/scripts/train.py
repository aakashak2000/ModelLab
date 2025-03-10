import argparse
import os
import sys
import time
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

class TextData:
    def __init__(self, file_path, seq_length=100, batch_size=32, char_level=True, train_ratio=0.9):
        self.file_path = file_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.char_level = char_level
        self.train_ratio = train_ratio

        print('Loading text from {file_path}...')
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        if char_level:
            self.tokens = list(self.text)
        else:
            self.tokens = self.text.split()
        
        self.chars = sorted(list(set(self.tokens)))
        self.vocab_size = len(self.chars)
        print('Vocab Size:', self.vocab_size)

        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        self.encoded_text = [self.char_to_idx[token] for token in self.tokens]

        train_size = int(len(self.encoded_text) * train_ratio)
        self.train_data = self.encoded_text[:train_size]
        self.val_data = self.encoded_text[train_size:]

        print(f"Total tokens: {len(self.encoded_text)}")
        print(f"Training tokens: {len(self.train_data)}")
        print(f"Validation tokens: {len(self.val_data)}")

    def get_batch(self, split='train'):
        if split not in ['train', 'val']:
            raise ValueError(f"'split': Expected 'train' or 'val', got {split}")
            return
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(0, len(data) - self.seq_length - 1, (self.batch_size,))

        x = torch.stack([torch.tensor(data[i: i + self.seq_length]) for i in ix])
        y = torch.stack([torch.tensor(data[i + 1: i + self.seq_length + 1]) for i in ix])

        return x, y
    
    def decode(self, indices):

        tokens = [self.idx_to_char[ix] for ix in indices]
        if self.char_level:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)    
        
    def save_vocab(self, vocab_path):
        vocab_info = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'char_level': self.char_level
        }
        torch.save(vocab_info, vocab_path)
        print(f"Vocabulary Saved to {vocab_path}")

def train(model, data, learning_rate=0.001, num_epochs=10, eval_interval=500, save_path=None, vocab_path=None, device='cpu'):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float("inf")

    tokens_per_epoch = len(data.train_data)
    total_iters = num_epochs * (tokens_per_epoch // (data.batch_size * data.seq_length))

    print(f"Starting training on {device}...")
    print(f"Estimated total iterations: {total_iters}")
    
    iter_num = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        batches_per_epoch = max(1, tokens_per_epoch // (tokens_per_epoch // (data.batch_size * data.seq_length)))
        for _ in range(batches_per_epoch):
            x_batch, y_batch = data.get_batch('train')
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits, _ = model(x_batch)
            logits = logits.view(-1, logits.size(-1))
            targets = y_batch.view(-1)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            history['train_loss'].append(loss.item())
            
            if iter_num % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    x_val, y_val = data.get_batch('val')
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    val_logits, _ = model(x_val)

                    val_logits = val_logits.view(-1, val_logits.size(-1))
                    val_targets = y_val.view(-1)

                    val_loss = criterion(val_logits, val_targets)
                    history['val_loss'].append(val_loss.item())

                    train_perplexity = torch.exp(torch.tensor(loss.item())).item()
                    val_perplexity = torch.exp(torch.tensor(val_loss.item())).item()
                    elapsed = time.time() - start_time

                    print(f"Epoch {epoch+1}/{num_epochs}, Iter {iter_num}/{total_iters}, \n" 
                          f"Train Loss: {loss.item():.4f}, Train Perplexity: {train_perplexity:.2f}, \n"
                          f"Val Loss: {val_loss.item():.4f}, Val Perplexity: {val_perplexity:.2f}, \n"
                          f"Time: {elapsed:.2f}s\n")
                    
                    if iter_num > 0:
                        seed_indices = x_val[0, :10].tolist()
                        seed_text = data.decode(seed_indices)
                        print(f'Seed text sample: {seed_text}')                    

                    if val_loss.item() < best_val_loss and save_path:
                        best_val_loss = val_loss.item()
                        print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
                        
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss.item(),
                            'epoch': epoch,
                            'model_config': {
                                'vocab_size': model.vocab_size,
                                'embedding_dim': model.embedding_dim,
                                'hidden_dim': model.hidden_dim,
                                'num_layers': model.num_layers
                            }
                        }, save_path)
                        if vocab_path:
                            data.save_vocab(vocab_path)
                model.train()
            iter_num += 1
                    

def main():
    parser = argparse.ArgumentParser(description='Train a language model on text data')    
    ## Model Arguments
    parser.add_argument('--model', type=str, required=True, 
                        help='Model Architecture to use from the following:\n\t1. \n\t1.rnn \n\t2.lstm \n\t3.gru \n\t4.seq2seq \n\t5.attention \n\t6.transformer \n\t7.miniGPT')
    ## Data Arguments
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to text file for training')
    parser.add_argument('--char-level', action='store_true', default=True,
                        help='Use character-level tokenization')
    parser.add_argument('--seq-length', type=int, default=100,
                        help='Sequence length for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch Size for training')
    
    ## Training Arguments
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for optimiser')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--eval-interval', type=int, default=500,
                        help='Evaluate model every N iterations')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Dimension of word embeddings')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Dimension of hidden state')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')

    ## Other arguments
    parser.add_argument('--model-name', type=str, default=None,
                        help='Custom name for the saved model file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run on')
    
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file} does not exist.")
        return
    models_dir = project_root / 'src' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name if args.model_name else args.model
    model_save_path = models_dir / f"{model_name}_best.pt"
    vocab_save_path = models_dir / f"{model_name}_vocab.pt"

    data = TextData(
        file_path = args.data_file,
        seq_length = args.seq_length,
        batch_size = args.batch_size,
        char_level = args.char_level
    )

    try:
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
        module = importlib.import_module(model_path)
        model_class = getattr(module, model_class_name)
        print(f'Successfully imported {model_class_name} from {model_path}')
    except Exception as e:
        print(f"Error importing model: {e}")
        print(f"Make sure the model exists at src/architectures/{args.model}/{args.model}.py")
        # print(f"and contains a class named {model_class_name}")
        return
    
    print('Creating new model')
    model = model_class(
        vocab_size=data.vocab_size,
        embedding_dim = args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers = args.num_layers,
        dropout = args.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    history = train(
        model = model,
        data = data,
        learning_rate = args.learning_rate,
        num_epochs = args.num_epochs,
        eval_interval = args.eval_interval,
        save_path = model_save_path,
        vocab_path = vocab_save_path,
        device = args.device
    )
    print(f'Model saved to {model_save_path}')
    print('DONE!')

if __name__ == '__main__':
    main()