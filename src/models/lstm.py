import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

    def forward(self, input_seqs):
        lstm_outputs, lstm_hiddens = self.lstm(input_seqs)
        outputs = self.linear(lstm_outputs)
        return outputs

