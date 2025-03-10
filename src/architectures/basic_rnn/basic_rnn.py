import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, nonlinearity='tanh', dropout=0.0, batch_first=True):
        super(BasicRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          nonlinearity=nonlinearity, 
                          dropout=dropout, 
                          batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):

        if hidden is None:
            batch_size = x.size(0) if self.batch_first else x.size(1) # x -> (batch_size, seq_length, input_size) if batch_first else (seq_length, batch_size, input_size)
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device) # hidden -> (num_layers, batch_size, hidden_size)

        output, hidden = self.rnn(x, hidden)

        if self.batch_first:
            last_output = output[:, -1, :]
        else:
            last_output = output[-1, :, :]

        output = self.fc(last_output) 
        return output, hidden
    

    def generate(self, initial_input, steps, temperature=1.0):
        
        self.eval()
        current_input = initial_input
        hidden = None
        generated_sequence = []

        with torch.no_grad():
            for _ in range(steps):
                output, hidden = self(current_input, hidden)
                if temperature != 1.0:
                    output = output / temperature
                probabilities = torch.Softmax(output, dim=-1) # output -> (batch_size, output_size)
                predicted = torch.multinomial(probabilities, 1)
                generated_sequence.append(predicted)
                current_input = torch.zeros(initial_input.size(0), 1, self.input_size, device=initial_input.device)

                indices = predicted.view(-1).tolist()
                for i, idx in enumerate(indices):
                    current_input[i, 0, idx] = 1

        return torch.cat(generated_sequence, dim=1) 
    
class CharRNN(BasicRNN):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, dropout=0):
        super(CharRNN, self).__init__(
            input_size=embedding_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            num_layers=num_layers,
            dropout=dropout
        )

        self.embedding = nn.Embedding(vocab_size, embedding_size)

    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        return super(CharRNN, self).forward(embedded, hidden)
    
    def generate_text(self, initial_text, char_to_idx, idx_to_char, max_length=100, temperature=1.0):
        self.eval()

        initial_indices = [char_to_idx.get(char, 0) for char in initial_text]
        current_input = torch.tensor(initial_indices, dtype=torch.long).unsqueeze(0)

        generated_text = initial_text
        hidden = None
        with torch.no_grad():
            for _ in range(max_length):
                embedded = self.embedding(current_input[:, -1:])
                output, hidden = self.rnn(embedded, hidden)
                output = self.fc(output.squeeze(1))
                if temperature != 1.0:
                    output = output/temperature

                probabilities = torch.softmax(output, dim=-1)
                predicted_idx = torch.multinomial(probabilities, 1).item()
                predicted_char = idx_to_char.get(predicted_idx, '.')
                generated_text += predicted_char
                current_input = torch.tensor([[predicted_idx]], dtype = torch.long)

        return generated_text
