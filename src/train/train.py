import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()

def train(input_batches, target_batches, decoder_feature_batches, encoder, decoder, encoder_optimizer,
          decoder_optimizer):

    input_batches = torch.from_numpy(input_batches)
    target_batches = torch.from_numpy(target_batches)
    decoder_feature_batches = torch.from_numpy(decoder_feature_batches)

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    # Get some constants
    input_lengths = input_batches.shape[0]
    target_lengths = target_batches.shape[0]
    batch_size = input_batches.shape[1]

    # Prepare input and output variables
    go = torch.zeros(1, batch_size, decoder.output_size)
    decoder_inputs = torch.cat((go, target_batches[:-1]), dim=0)
    input_batches = Variable(input_batches, requires_grad=False)
    target_batches = Variable(target_batches, requires_grad=False)
    decoder_inputs = Variable(torch.cat((decoder_inputs, decoder_feature_batches), dim=2), requires_grad=False)
    all_decoder_outputs = Variable(torch.zeros(target_lengths, batch_size, decoder.output_size), requires_grad=False)

    # Send variables to cuda
    if use_cuda:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()
        decoder_inputs = decoder_inputs.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run encoder
    encoder_outputs, encoder_hiddens = encoder(input_batches, None)

    # Get decoder initial states
    decoder_hidden_state = encoder_hiddens[0]
    decoder_cell_state = torch.zeros_like(decoder_hidden_state)

    # Run through decoder one time step at a time
    for t in range(target_lengths):
        decoder_output, (decoder_hidden_state, decoder_cell_state) = decoder(
            decoder_inputs[[t]], (decoder_hidden_state, decoder_cell_state)
        )
        all_decoder_outputs[t] = decoder_output

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(all_decoder_outputs.transpose(0, 1).contiguous(),
                     target_batches.transpose(0, 1).contiguous())

    # Update parameters
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]

def evaluate(input_batches, target_batches, decoder_features, encoder, decoder):
    input_batches = torch.from_numpy(input_batches)
    target_batches = torch.from_numpy(target_batches)
    decoder_features = torch.from_numpy(decoder_features)

    batch_size = input_batches.shape[1]
    output_seq_len = target_batches.shape[0]

    go = torch.zeros(1, batch_size, decoder.output_size)
    decoder_input = torch.cat((go, decoder_features[[0]]), dim=2)

    input_batches = Variable(input_batches, requires_grad=False)
    target_batches = Variable(target_batches, requires_grad=False)
    decoder_features = Variable(decoder_features, requires_grad=False)
    decoder_input = Variable(decoder_input, requires_grad=False)
    all_decoder_outputs = Variable(torch.zeros(output_seq_len, batch_size, decoder.output_size), requires_grad=False)

    if use_cuda:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()
        decoder_features = decoder_features.cuda()
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run encoder
    encoder_outputs, encoder_hiddens = encoder(input_batches, None)

    # Get decoder initial states
    decoder_hidden_state = encoder_hiddens[0]
    decoder_cell_state = torch.zeros_like(decoder_hidden_state)

    for i in range(output_seq_len):
        decoder_output, (decoder_hidden_state, decoder_cell_state) = decoder(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        all_decoder_outputs[i] = decoder_output

        if i != output_seq_len - 1:
            decoder_input = Variable(torch.cat((decoder_output.data, decoder_features[[i]].data), dim=2), requires_grad=False)
            decoder_input = decoder_input if use_cuda else decoder_input

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(all_decoder_outputs.transpose(0, 1).contiguous(),
                         target_batches.transpose(0, 1).contiguous())

    return all_decoder_outputs.data













