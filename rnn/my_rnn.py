import torch
import torch.nn as nn

class MyRNN(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        embedding_dim = 32
        rnn_hidden_size = 128
        fc_size = 128
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
            embedding_dim=embedding_dim, padding_idx=0) 
        # layer: long-short term memory
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden_size,
                num_layers=2, batch_first=True)
        # fully-connected layer
        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features=fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        # Packs a Tensor containing padded sequences of variable length.
        lengths = lengths.cpu().numpy()
        out = nn.utils.rnn.pack_padded_sequence(out, lengths, \
            batch_first=True, enforce_sorted=False)
        # 
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

   
