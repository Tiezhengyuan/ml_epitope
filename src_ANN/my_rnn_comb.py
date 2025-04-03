import torch
import torch.nn as nn

class MyRnnComb(nn.Module):
    
    def __init__(self, vocab_size:int, embedding_dim:int, num_features:int=None):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = 0
        ) 
        # RNN: long-short term memory
        hidden_size = 64 # hidden state is 32 at each step
        self.rnn = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = 2,
            batch_first = True
        )
        self.rnn_linear = nn.Sequential(
            nn.Linear(
                in_features = hidden_size,
                out_features = 1024
            ),
            nn.ReLU(),
        )
        # ANN for features
        self.ann = nn.Sequential(
            nn.Linear(
                in_features = num_features,
                out_features = 256
            ),
            nn.ReLU(),
        )
        # combinaton: fully-connected layers
        self.linear = nn.Sequential(
            nn.Linear(
                in_features = 1024 + 256,
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
        input: batch_size x num_features
        lengths: length = batch_size
        '''
        # print(input)
        # batch_size x num_features x embedding_dim
        input_text = self.embedding(input['texts'])
        # Packs a Tensor containing padded sequences of variable length.
        lengths = input['lengths'].cpu().numpy()
        # RNN
        # output: sum of lengths x embedding_dim
        input_text = nn.utils.rnn.pack_padded_sequence(
            input = input_text,
            lengths = lengths,
            batch_first = True,
            enforce_sorted = False
        )
        # out: sum of lengths x hidden_size
        out_rnn, (hidden, cell) = self.rnn(input_text)
        # use the last hidden state
        # shape: (batch_size, hidden_size) -> (32, 64)
        out_rnn = hidden[-1, :, :]
        # print(out_rnn.shape)
        # RNN linear
        out_rnn = self.rnn_linear(out_rnn)

        # ANN
        # features for ANN
        input_features = input['features']
        # print(input_features.shape)
        out_ann = self.ann(input_features)

        # combination: fully-connected
        out = torch.hstack([out_rnn, out_ann])
        out = self.linear(out)
        return out
