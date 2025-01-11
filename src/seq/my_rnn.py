import torch
import torch.nn as nn

class MyRNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = 0
        ) 
        # layer: long-short term memory
        hidden_size = 64 # hidden state is 32 at each step
        self.rnn = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = 2,
            batch_first = True
        )
        # fully-connected layers
        self.linear = nn.Sequential(
            nn.Linear(
                in_features = hidden_size,
                out_features = 128
            ),
            nn.ReLU(),
            nn.Linear(
                in_features = 128,
                out_features = 1
            ),
            nn.Sigmoid(),
        )

    def forward(self, input:dict):
        '''
        text: batch_size x num_features
        lengths: length = batch_size
        '''
        # batch_size x num_features x embedding_dim
        input = self.embedding(input['texts'])

        # Packs a Tensor containing padded sequences of variable length.
        lengths = input['lengths'].cpu().numpy()
        # output: sum of lengths x embedding_dim
        input = nn.utils.rnn.pack_padded_sequence(
            input = input,
            lengths = lengths,
            batch_first = True,
            enforce_sorted = False
        )
        # out: sum of lengths x hidden_size
        out, (hidden, cell) = self.rnn(input)

        # use the last hidden state
        out = hidden[-1, :, :]
        # fully-connected
        out = self.linear(out)
        return out
