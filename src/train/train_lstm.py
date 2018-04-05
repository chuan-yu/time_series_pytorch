import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def train(input_batches, target_batches, lstm, optimizer, use_cuda=False):
    optimizer.zero_grad()

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

def evaluate(inputs, targets, lstm, use_cuda=False):
    n_batches = inputs.shape[0]
    total_loss = 0
    for i in range(n_batches):
        input_batch = inputs[i]
        target_batch = targets[i]
        prediction_batch = predict(input_batch, lstm, use_cuda)
        loss = get_mse(prediction_batch, target_batch)
        total_loss += loss

    loss_mean = total_loss / n_batches

    return loss_mean

def get_mse(predictions, targets):
    predictions = predictions.view(predictions.numel())
    targets = targets.view(targets.numel())
    mse = ((predictions - targets) ** 2).mean()
    return mse


def predict(input_sequences, lstm, use_cuda=False):

    seq_len = input_sequences.shape[0]
    batch_size = input_sequences.shape[1]

    input_sequences = Variable(input_sequences, requires_grad=False)
    all_predicitons = Variable(torch.zeros(seq_len, batch_size, 1))
    if use_cuda:
        input_sequences = input_sequences.cuda()
        all_predicitons = all_predicitons.cuda()

    x = input_sequences[[0]]
    for i in range(seq_len):
        predictions = lstm(x)
        all_predicitons[i] = predictions
        if i < seq_len - 1:
            features_next = input_sequences[i + 1, :, 1:].unsqueeze(0)
            x = torch.cat((predictions, features_next), dim=2)

    return all_predicitons.data


def predict_batches(input_batches, lstm, use_cuda=False):
    n_batches = input_batches.shape[0]
    total_loss = 0
    all_predictions = torch.zeros(n_batches, input_batches.shape[1], input_batches.shape[2], 1)
    for i in range(n_batches):
        input_batch = input_batches[i]
        prediction_batch = predict(input_batch, lstm, use_cuda)
        all_predictions[i] = prediction_batch

    return all_predictions















