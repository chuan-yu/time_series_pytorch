import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def train(input_batches, target_batches, lstm, optimizer, use_cuda=False):

    lstm = lstm.train()
    optimizer.zero_grad()

    input_batches = Variable(input_batches, requires_grad=False)
    target_batches = Variable(target_batches, requires_grad=False)

    if use_cuda:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()

    prediction_batches = lstm(input_batches)
    prediction_batches = prediction_batches[-1].transpose(0, 1)
    prediction_batches = prediction_batches.unsqueeze(2)

    criterion = nn.MSELoss()

    loss = criterion(prediction_batches.transpose(0, 1).contiguous(),
                   target_batches.transpose(0, 1).contiguous())

    loss.backward()
    optimizer.step()

    return loss.data[0]

def evaluate(inputs, targets, lstm, use_cuda=False):
    lstm = lstm.eval()
    if use_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
    n_batches = inputs.shape[0]
    total_loss = 0
    for i in range(n_batches):
        lstm.hidden = None
        input_batch = inputs[i]
        target_batch = targets[i]
        prediction_batch = predict(input_batch, lstm, use_cuda)
        loss = get_mse(prediction_batch, target_batch)
        total_loss += loss

    loss_mean = total_loss / n_batches

    return loss_mean

def get_mse(predictions, targets):
    predictions = predictions.contiguous().view(predictions.numel())
    targets = targets.contiguous().view(targets.numel()).contiguous()
    mse = ((predictions - targets) ** 2).mean()
    return mse


def predict(input_sequences, lstm, use_cuda=False):
    lstm = lstm.eval()
    input_sequences = Variable(input_sequences, requires_grad=False)
    if use_cuda:
        input_sequences = input_sequences.cuda()
    predictions = lstm(input_sequences)
    predictions = predictions[-1].transpose(0, 1).unsqueeze(-1)
    return predictions.data


def predict_batches(input_batches, target_batches, lstm, use_cuda=False):
    if use_cuda:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()
    lstm = lstm.eval()
    n_batches = input_batches.shape[0]
    output_size = lstm.output_size
    batch_size = input_batches.shape[2]
    all_predictions = torch.zeros(n_batches, output_size, batch_size, 1)
    total_loss = 0
    for i in range(n_batches):
        lstm.hidden = None
        input_batch = input_batches[i]
        target_batch = target_batches[i]
        prediction_batch = predict(input_batch, lstm, use_cuda)
        all_predictions[i] = prediction_batch
        loss = get_mse(prediction_batch, target_batch)
        total_loss += loss

    mean_loss = total_loss / n_batches

    return all_predictions, mean_loss