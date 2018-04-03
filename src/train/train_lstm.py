import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def train(input_batches, target_batches, lstm, optimizer, use_cuda=False):
    input_batches = torch.from_numpy(input_batches)
    target_batches = torch.from_numpy(target_batches)

    optimizer.zero_grad()

    input_length = input_batches.shape[0]
    batch_size = input_batches.shape[1]

    input_batches = Variable(input_batches, requires_grad=False)
    target_batches = Variable(target_batches, requires_grad=False)



    if use_cuda:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()

    prediction_batches = lstm(input_batches)

    criterion = nn.MSELoss()

    loss = criterion(prediction_batches.transpose(0, 1).contiguous(),
                   target_batches.transpose(0, 1).contiguous())

    loss.backward()
    optimizer.step()

    return loss.data[0]






