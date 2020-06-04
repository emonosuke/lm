import torch
import torch.nn as nn


class RNNLM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout=0.2, tie_weights=False):
        super(RNNLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                           batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            if hidden_size != emb_size:
                raise ValueError("When tie_weight=True, hidden_size must be equal to emb_size")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, hidden):
        embedded = self.drop(self.encoder(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
