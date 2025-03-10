# scripts/train_model.py

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys

# Add project root to path
#sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
print(os.listdir(sys.path[-1]))
from models.basic_rnn.basic_rnn import CharRNN
from utils.preprocessing import read_text_file, create_char_mappings, preprocess_data, create_dataloaders

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path=None):
    """Train the model and validate."""
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': []
    }
    
    for epoch in range(num_epochs):

        print(f"{epoch} of {num_epochs}")
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Reshape for loss calculation
            # outputs: [batch_size, seq_len, vocab_size]
            #outputs = outputs.view(-1, outputs.size(-1))
            #targets = targets.view(-1)
            last_targets = targets[:, -1]
            
            # Calculate loss
            loss = criterion(outputs, last_targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs, _ = model(inputs)
                #outputs = outputs.view(-1, outputs.size(-1))
                #targets = targets.view(-1)
                last_targets = targets[:, -1]
                
                loss = criterion(outputs, last_targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        # Save training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['epoch_times'].append(epoch_time)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s')
        
        # Save the best model
        if save_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(save_path, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_save_path)
            print(f'Model saved to {model_save_path}')
    
    # Save training history
    if save_path:
        with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
            json.dump(training_history, f)
    
    return training_history

def main():
    parser = argparse.ArgumentParser(description='Train a character-level RNN model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--output_dir', type=str, default='models/basic_rnn', help='Directory to save the model')
    parser.add_argument('--sequence_length', type=int, default=100, help='Sequence length for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_size', type=int, default=128, help='Size of character embeddings')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Read and preprocess data
    print(f'Reading data from {args.input_file}...')
    text = read_text_file(args.input_file)
    print(f'Text length: {len(text)} characters')
    
    # Create mappings
    char_to_idx, idx_to_char, vocab_size = create_char_mappings(text)
    print(f'Vocabulary size: {vocab_size} characters')
    
    # Save mappings
    mappings = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }
    with open(os.path.join(args.output_dir, 'char_mappings.json'), 'w') as f:
        json.dump(mappings, f)
    
    # Preprocess data
    print(f'Preprocessing data with sequence length {args.sequence_length}...')
    inputs, targets = preprocess_data(text, char_to_idx, args.sequence_length)
    print(f'Created {len(inputs)} sequences')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(inputs, targets, args.batch_size)
    print(f'Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}')
    
    # Initialize model
    model = CharRNN(
        vocab_size=vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    print('Starting training...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_path=args.output_dir
    )
    
    print('Training complete!')

if __name__ == '__main__':
    main()
