import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.embeddings(src)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(input_size=output_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers>1 else 0.0,
                            batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.embeddings(input)
        output, hidden = self.lstm(embedded)
        output = output.squeeze(1)
        prediction = self.fc_out(output)
        return prediction, hidden
    

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(Seq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, tgt=None, teacher_forcing_ratio = 0.5):
        pass