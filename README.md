# ModelLab: Evolution of Text Generation Models

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NLP-8A2BE2?style=for-the-badge&logo=nlp&logoColor=white" alt="NLP" />
  <img src="https://img.shields.io/badge/Text_Generation-FF6F61?style=for-the-badge&logo=&logoColor=white" alt="Text Generation" />
  <br />
  <img src="https://img.shields.io/github/license/aakashak2000/modellab" alt="License" />
  <img src="https://img.shields.io/github/stars/aakashak2000/modellab" alt="Stars" />
  <img src="https://img.shields.io/github/forks/aakashak2000/modellab" alt="Forks" />
  <img src="https://img.shields.io/github/issues/aakashak2000/modellab" alt="Issues" />
</div>

<div align="center">
  <h3>A comprehensive toolkit for understanding and building text generation models</h3>
  <p>From RNNs to LSTMs, explore the evolution of neural architectures for text generation</p>
</div>

---

## 📜 Overview

ModelLab is an educational and research project that offers a hands-on journey through the evolution of text generation models. The repository provides clean implementations of various neural network architectures with a unified interface for training, experimentation, and generation.

The project serves as both a learning resource for those new to NLP and a research toolkit for experimenting with different model architectures.

<div align="center">
  <img src="https://raw.githubusercontent.com/aakashak2000/modellab/main/assets/architecture_comparison.png" alt="Architecture Comparison" width="80%" />
</div>

## 🌟 Key Features

- **Multiple Architectures**: Clean implementations of RNN and LSTM models with unified interfaces
- **Unified Training Pipeline**: Common framework for consistent training across architectures
- **Flexible Text Generation**: Temperature-controlled sampling for diverse text generation
- **Easy Experimentation**: Notebooks for quick experiments and visualizations
- **Extensible Framework**: Designed for easy addition of new architectures and techniques

## 🏗️ Project Structure

```
modellab/
├── src/                    # Main source code
│   ├── architectures/      # Model implementations
│   │   ├── rnn/            # RNN architectures
│   │   └── lstm/           # LSTM architectures
│   ├── models/             # Model training interface
│   ├── data/               # Data loading and processing
│   ├── utils/              # Utility functions
│   └── scripts/            # Training and generation scripts
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── data/                   # Raw and processed datasets
└── results/                # Generated samples and model metrics
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/modellab.git
cd modellab

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

```bash
# Train an LSTM model on a text dataset
python src/scripts/train.py --model lstm \
                           --data-file data/your_text_file.txt \
                           --seq-length 100 \
                           --batch-size 32 \
                           --num-epochs 10 \
                           --learning-rate 0.001
```

### Generating Text

```bash
# Generate text using a trained model
python src/scripts/generate.py --model lstm \
                              --model-path src/models/lstm_best.pt \
                              --seed-text "Once upon a time" \
                              --max-length 500 \
                              --temperature 0.8
```

## 💡 Examples

### LSTM Text Generation

```python
from src.architectures.lstm.lstm import LSTM

model = LSTM(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2
)

# Generate 100 tokens starting with given tokens
generated = model.generate(
    initial_tokens=[5, 10, 15], 
    max_length=100,
    temperature=0.8
)
```

## 📊 Performance Comparison

| Model | Parameters | Perplexity | Training Time | Inference Time |
|-------|------------|------------|--------------|----------------|
| RNN   | 2.5M       | 102.3      | 1.0x         | 1.0x           |
| LSTM  | 3.8M       | 84.7       | 1.3x         | 1.2x           |

## 📓 Notebooks

- `01_data_exploration.ipynb`: Explore and understand the training data
- `02_model_comparison.ipynb`: Compare performance of different architectures 
- `03_generation_showcase.ipynb`: Interactive text generation examples

## 📚 Learning Resources

- [Understanding RNNs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

## 🛣️ Roadmap

- [x] RNN implementation
- [x] LSTM implementation
- [ ] GRU implementation
- [ ] Seq2Seq with attention
- [ ] Transformer architecture
- [ ] Fine-tuning capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for inspiration and educational resources
- [PyTorch Team](https://github.com/pytorch/pytorch) for the amazing deep learning framework
- All contributors and users of this project

---

<div align="center">
  <p>If you find ModelLab useful, please consider giving it a ⭐!</p>
  <p>Made with ❤️ by <a href="https://github.com/yourusername">Your Name</a></p>
</div>
