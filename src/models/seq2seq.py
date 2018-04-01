import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)

    def forward(self, input_seqs, hidden=None):
        outputs, hiddens = self.lstm(input_seqs, hidden)
        return outputs, hiddens

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.dense_out = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, input, hidden):
        outputs, hiddens = self.lstm(input, hidden)
        outputs = self.dense_out(outputs)
        return outputs, hiddens


