# Source Code Documentation

This directory contains the core implementation of the "Evolution of Text Generation Models" project. The codebase demonstrates the progression of neural text generation models from simple RNNs to more complex architectures like Transformers and GPT-like models.

## Directory Structure

```
src/
├── architectures/       # Neural network model architectures
│   ├── rnn/             # Recurrent Neural Networks
│   ├── lstm/            # Long Short-Term Memory Networks
│   ├── gru/             # Gated Recurrent Units
│   ├── seq2seq/         # Sequence-to-Sequence models
│   ├── attention/       # Attention mechanisms
│   ├── transformer/     # Transformer architecture
│   └── miniGPT/         # GPT-style model implementation
├── data/                # Data loading and processing utilities
├── models/              # Model training interfaces and saved models
├── scripts/             # Training and generation scripts
└── utils/               # Utility functions and helper methods
```

## Architecture Implementation

Each architecture subfolder contains at least two implementations:
- `{model_name}.py`: PyTorch implementation using built-in modules
- `{model_name}_from_scratch.py`: Implementation from first principles to demonstrate core concepts

### Current Implementation Status

| Architecture | Status | Notes |
|--------------|--------|-------|
| RNN          | ✅ Complete | Basic recurrent neural network implementation |
| LSTM         | ✅ Complete | Long Short-Term Memory network |
| GRU          | 🚧 In Progress | Gated Recurrent Unit implementation |
| Seq2Seq      | 🚧 In Progress | Sequence-to-sequence model |
| Attention    | 🚧 In Progress | Attention mechanism implementation |
| Transformer  | 🚧 In Progress | Full transformer architecture |
| miniGPT      | 🚧 In Progress | Smaller GPT-style model |

## Key Components

### Data Module (`data/`)

Handles data processing for text generation tasks, including:
- Text loading and preprocessing
- Tokenization (character and word-level)
- Batching and sequence preparation
- Vocabulary management

### Models Module (`models/`)

Contains interfaces for model training and saved model artifacts:
- Model configuration specifications
- Trained model checkpoints
- Vocabulary files

### Scripts (`scripts/`)

Provides command-line utilities for model training and text generation:

#### Training Script (`train.py`)

```bash
python src/scripts/train.py --model lstm --data-file data/your_text.txt --num-epochs 10
```

Key parameters:
- `--model`: Architecture to use (rnn, lstm, gru, etc.)
- `--data-file`: Text file for training
- `--char-level`: Use character-level tokenization (default: True)
- `--seq-length`: Sequence length for training (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--num-epochs`: Training epochs (default: 10)
- `--embedding-dim`: Dimension of embeddings (default: 128)
- `--hidden-dim`: Dimension of hidden state (default: 256)
- `--num-layers`: Number of layers (default: 2)
- `--dropout`: Dropout probability (default: 0.2)

#### Generation Script (`generate.py`)

```bash
python src/scripts/generate.py --model lstm --model-path src/models/lstm_best.pt --seed-text "Once upon a time"
```

Key parameters:
- `--model`: Architecture to use
- `--model-path`: Path to trained model file
- `--seed-text`: Starting text for generation
- `--num-samples`: Number of samples to generate (default: 5)
- `--max-length`: Maximum length of generated text (default: 500)
- `--temperature`: Sampling temperature (default: 0.8)

### Utilities (`utils/`)

Contains helper functions for:
- Logging and visualization
- Performance metrics
- Model evaluation tools
- Text processing utilities

## Model Details

### RNN Implementation

The basic RNN implementation demonstrates:
- Embedding layer for token representation
- Recurrent layer for sequential processing
- Linear output layer for next token prediction
- Temperature-based sampling for text generation

### LSTM Implementation

The LSTM model extends the RNN with:
- Input, forget, and output gates
- Cell state for long-term memory
- Advanced text generation capabilities
- Improved handling of long-range dependencies

## Usage Examples

### Training a New Model

```python
from src.architectures.lstm.lstm import LSTM
from src.scripts.train import TextData, train

# Load and prepare data
data = TextData(
    file_path="data/your_text.txt",
    seq_length=100,
    batch_size=32,
    char_level=True
)

# Create model
model = LSTM(
    vocab_size=data.vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.2
)

# Train the model
history = train(
    model=model,
    data=data,
    learning_rate=0.001,
    num_epochs=10,
    save_path="src/models/lstm_custom.pt"
)
```

### Generating Text with a Trained Model

```python
import torch
from src.architectures.lstm.lstm import LSTM
from src.scripts.generate import generate_text, load_vocab

# Load vocabulary
vocab_info = load_vocab("src/models/lstm_vocab.pt")
char_to_idx = vocab_info['char_to_idx']
idx_to_char = vocab_info['idx_to_char']

# Load model
checkpoint = torch.load("src/models/lstm_best.pt")
model_config = checkpoint['model_config']

model = LSTM(
    vocab_size=model_config['vocab_size'],
    embedding_dim=model_config['embedding_dim'],
    hidden_dim=model_config['hidden_dim'],
    num_layers=model_config['num_layers']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Generate text
generated_text = generate_text(
    model=model,
    seed_text="The quick brown fox",
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    max_length=500,
    temperature=0.8
)
print(generated_text)
```

## Contributing

When adding new architectures:
1. Create a new folder in `architectures/` with your model name
2. Implement both standard and from-scratch versions
3. Ensure the model class has:
   - Compatible initialization parameters with other models
   - `forward()` method returning outputs and hidden states
   - `generate()` method for text generation

## Future Development

Planned enhancements:
- Complete implementation of advanced architectures (Transformer, miniGPT)
- Add fine-tuning capabilities for pre-trained models
- Implement beam search and other advanced decoding strategies
- Add model evaluation metrics beyond perplexity
