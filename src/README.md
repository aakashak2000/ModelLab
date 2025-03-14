# ModelLab Source Code

This directory contains the core implementation of the ModelLab project, organized to provide a clear structure for different components of the text generation models.

## Directory Structure

### 🏛️ `architectures/`
Neural network architectures for text generation, arranged by model type:

- **`rnn/`**: Recurrent Neural Networks
  - `rnn.py`: PyTorch implementation using nn.RNN
  - `rnn_from_scratch.py`: Custom implementation (WIP)

- **`lstm/`**: Long Short-Term Memory Networks
  - `lstm.py`: PyTorch implementation using nn.LSTM
  - `lstm_from_scratch.py`: Custom implementation (WIP)

- **`gru/`**: Gated Recurrent Units
  - `gru.py`: PyTorch implementation ✓
  - `gru_from_scratch.py`: Custom implementation (WIP)

- **`seq2seq/`**: Sequence-to-Sequence Models
  - `seq2seq.py`: Encoder-decoder architecture (encoder-decoder blocks implemented)
  - `seq2seq_from_scratch.py`: Custom implementation (WIP)

- **`attention/`**: Attention Mechanisms
  - `attention.py`: Attention-enhanced models (WIP)
  - `attention_from_scratch.py`: Custom implementation (WIP)

- **`transformer/`**: Transformer Architecture
  - `transformer.py`: Implementation based on "Attention is All You Need" (WIP)
  - `transformer_from_scratch.py`: Custom implementation (WIP)

- **`miniGPT/`**: GPT-like Models
  - `miniGPT.py`: Lightweight GPT implementation (WIP)
  - `miniGPT_from_scratch.py`: Custom implementation (WIP)

### 📊 `models/`
Model training interfaces and saved model checkpoints:
- Training configuration
- Model serialization/deserialization
- Common training utilities

### 📁 `data/`
Data processing utilities:
- Text data loading
- Tokenization and preprocessing
- Vocabulary management
- Dataset creation

### 🔧 `utils/`
Common utility functions:
- Logging
- Metrics calculation
- Visualization helpers
- Text processing tools

### 📜 `scripts/`
Executable scripts for training and generation:
- `train.py`: Universal training script for all model architectures
- `generate.py`: Text generation using trained models

## Implementation Guidelines

When implementing or extending model architectures, follow these guidelines:

1. **Interface Consistency**: All model classes should expose the same interface with `forward()` and `generate()` methods
2. **Documentation**: Include docstrings explaining the architecture, parameters, and usage
3. **Modular Design**: Implement complex models as composition of simpler components
4. **Code Reuse**: Leverage shared components in utils/ rather than duplicating functionality

## Usage Examples

### Training an LSTM Model

```python
from src.architectures.lstm.lstm import LSTM
from src.data.text_dataset import TextDataset
import torch

# Prepare data
dataset = TextDataset("path/to/data.txt", char_level=True)
data_loader = dataset.get_loader(batch_size=32, seq_length=100)

# Create model
model = LSTM(
    vocab_size=dataset.vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.2
)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for x_batch, y_batch in data_loader:
    output, _ = model(x_batch)
    loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Generating Text with a Trained Model

```python
from src.architectures.lstm.lstm import LSTM
import torch

# Load trained model
model = LSTM(vocab_size=128, embedding_dim=256, hidden_dim=512, num_layers=2)
checkpoint = torch.load("path/to/model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Generate text
seed_text = "Once upon a time"
seed_indices = [char_to_idx[c] for c in seed_text]  # Convert to indices
generated = model.generate(
    initial_tokens=seed_indices,
    max_length=200,
    temperature=0.8
)

# Convert indices back to text
generated_text = ''.join([idx_to_char[idx] for idx in generated])
print(generated_text)
```

## Development Status

| Architecture | PyTorch Implementation | From-Scratch Implementation | Status |
|--------------|:----------------------:|:---------------------------:|:------:|
| RNN          | ✅                     | ⏳                          | Active |
| LSTM         | ✅                     | ⏳                          | Active |
| GRU          | ✅                     | ⏳                          | Active |
| Seq2Seq      | 🔄                     | ⏳                          | In Progress |
| Attention    | ⏳                     | ⏳                          | Planned |
| Transformer  | ⏳                     | ⏳                          | Planned |
| MiniGPT      | ⏳                     | ⏳                          | Planned |

Legend:
- ✅ Complete
- 🔄 Partially Implemented
- ⏳ Planned

## Implementation Roadmap

1. **Current Phase**: Implement all architectures using PyTorch built-in modules
   - RNN, LSTM, GRU completed
   - Seq2seq encoder-decoder blocks implemented
   - Working on completing Seq2seq implementation

2. **Next Phase**: Implement all architectures from scratch using NumPy
   - Will start with RNN_from_scratch and progress through all models
   
3. **Final Phase**: Create notebooks to display results, metrics, and benchmarks
   - Compare performance across architectures
   - Visualize training metrics and generation capabilities
