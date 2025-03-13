import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim,
                 num_layers=1,
                 dropout=0.0):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout if num_layers>1 else 0,
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)

    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        embedded = self.embeddings(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)

        return output, hidden
    
    def generate(self, initial_tokens, max_length, temperature=1.0, device='cpu'):
        self.to(device)
        self.eval()

        current_tokens = torch.tensor([initial_tokens], dtype=torch.long, device=device)
        generated = initial_tokens.copy() if isinstance(initial_tokens, list) else initial_tokens.tolist()
        hidden = None
        with torch.no_grad():
            _, hidden = self(current_tokens, hidden)
            for _ in range(max_length):
                output, hidden = self(current_tokens, hidden)
                logits = output[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                current_tokens = torch.tensor([[next_token]], dtype=torch.long, device=device)
                generated.append(next_token)
        return generated




