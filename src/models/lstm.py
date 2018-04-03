import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.batch_size = input_size.shape[1]
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)))

    def forward(self, input_seqs, hidden=None):
        lstm_outputs, lstm_hiddens = self.lstm(input, self.hidden)
        outputs = self.linear(lstm_outputs)
        return outputs

