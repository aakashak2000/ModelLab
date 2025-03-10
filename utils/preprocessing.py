# utils/preprocessing.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def read_text_file(file_path):
    """Read a text file and return its content as a string."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_char_mappings(text):
    """Create character-to-index and index-to-character mappings."""
    # Get unique characters
    chars = sorted(list(set(text)))
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return char_to_idx, idx_to_char, len(chars)

def preprocess_data(text, char_to_idx, sequence_length=100):
    """
    Preprocess text data for character-level RNN training.
    
    Args:
        text (str): The input text
        char_to_idx (dict): Character to index mapping
        sequence_length (int): Length of input sequences
        
    Returns:
        inputs (list): List of input sequences
        targets (list): List of target sequences (next character)
    """
    # Convert text to indices
    text_indices = [char_to_idx[char] for char in text]
    
    # Create sequences
    inputs = []
    targets = []
    
    for i in range(0, len(text_indices) - sequence_length):
        # Input is a sequence of characters
        input_seq = text_indices[i:i + sequence_length]
        # Target is the next character after the sequence
        target_seq = text_indices[i + 1:i + sequence_length + 1]
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return inputs, targets

class TextDataset(Dataset):
    """Dataset for character-level text generation."""
    
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )

def create_dataloaders(inputs, targets, batch_size=64, train_ratio=0.8):
    """Create train and validation DataLoaders."""
    # Create dataset
    dataset = TextDataset(inputs, targets)
    
    # Split into train and validation
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader
