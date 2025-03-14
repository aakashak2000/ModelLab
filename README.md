# ModelLab: Evolution of Text Generation Models

ModelLab is a comprehensive project exploring the evolution of neural text generation architectures, from simple RNNs to complex transformer-based models. This repository provides implementations, training pipelines, and analysis tools to understand how different architectures approach the challenge of text generation.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT"/>
</p>

## 🚀 Project Overview

ModelLab demonstrates the progressive improvement of neural language models through the implementation of increasingly sophisticated architectures:

- **Basic RNN**: Simple recurrent networks with limited memory capabilities
- **LSTM**: Long Short-Term Memory networks that better handle long-range dependencies 
- **GRU**: Gated Recurrent Units, a more efficient alternative to LSTMs ✓
- **Seq2Seq**: Encoder-decoder architecture with partially implemented encoder-decoder blocks
- **Attention**: Models with attention mechanisms for better context awareness (WIP)
- **Transformer**: Modern architecture based on self-attention mechanisms (WIP)
- **MiniGPT**: Lightweight implementation of GPT-like architectures (WIP)

Each architecture is implemented both with PyTorch's built-in modules and from scratch to provide deeper insights into their inner workings.

## 📂 Project Structure

```
ModelLab/
├── src/                   # Main source code
│   ├── architectures/     # Neural network architectures
│   │   ├── rnn/           # RNN implementations
│   │   ├── lstm/          # LSTM implementations
│   │   ├── gru/           # GRU implementations (WIP)
│   │   ├── seq2seq/       # Sequence-to-sequence (WIP)
│   │   ├── attention/     # Attention mechanisms (WIP)
│   │   ├── transformer/   # Transformer architecture (WIP)
│   │   └── miniGPT/       # GPT-like model (WIP)
│   ├── models/            # Model training and interfaces
│   ├── data/              # Data loading and processing
│   ├── utils/             # Utility functions
│   └── scripts/           # Training and generation scripts
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── data/                  # Raw and processed datasets
└── results/               # Generated samples and metrics
```

## 🔧 Setup and Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ModelLab.git
cd ModelLab

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

```bash
# Basic RNN model training
python src/scripts/train.py --model rnn --data-file data/sample_text.txt --char-level --num-epochs 10

# LSTM model with custom parameters
python src/scripts/train.py \
    --model lstm \
    --data-file data/sample_text.txt \
    --char-level \
    --seq-length 128 \
    --batch-size 64 \
    --embedding-dim 256 \
    --hidden-dim 512 \
    --num-layers 2 \
    --dropout 0.3 \
    --num-epochs 20
```

### Generating Text

```bash
# Generate text using a trained model
python src/scripts/generate.py \
    --model lstm \
    --model-path src/models/lstm_best.pt \
    --seed-text "Once upon a time" \
    --max-length 500 \
    --temperature 0.8
```

## 📈 Model Comparison

The repository includes analysis tools to compare different architectures across metrics like:

- Training and validation loss
- Perplexity
- Generation quality
- Training efficiency
- Parameter count

Check the `notebooks/02_model_comparison.ipynb` notebook for detailed comparisons.

## 🔍 Example Outputs

Here are sample outputs from different model architectures (trained on the same dataset):

**RNN:**
```
The storee of the world was a little of the world of the story of the world...
```

**LSTM:**
```
The story begins with a character who finds themselves in an unusual situation...
```

## 📚 Learning Resources

This project is designed as a learning resource for understanding:

- The evolution of text generation architectures
- How improvements in neural network design address limitations
- Best practices for implementation and training
- Techniques for text generation, sampling, and evaluation

## 🤝 Contributing

Contributions are welcome! Areas that could use help:

- Implementing remaining architectures (Seq2Seq, Attention, Transformer, MiniGPT)
- Improving documentation and tutorials
- Adding visualization tools
- Enhancing training efficiency
- Creating more comprehensive evaluation metrics

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
