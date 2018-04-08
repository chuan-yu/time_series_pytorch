import os
from models.seq2seq import EncoderRNN, DecoderRNN
from train.train_s2s import train, evaluate, evaluate_batches
import torch
import reader
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

use_cuda = True
encoder_checkpoint = "../checkpoints/s2s/encoder.pt"
decoder_checkpoint = "../checkpoints/s2s/decoder.pt"

input_size = 4
output_size = 1
hidden_size = 200
n_layers = 3
lr = 0.001
batch_size = 32
n_epochs = 300
input_seq_len = 50
output_seq_len = 10
dropout = 0.5

HOLIDAYS = ['2016-03-25']
stations = [0, 8, 27, 32, 69, 75, 100, 110, 111]

for s in stations:
    print("station %i" % s)

    encoder_checkpoint = "../checkpoints/s2s/encoder/encoder-station-%s.pt" % str(s)
    decoder_checkpoint = "../checkpoints/s2s/decoder/decoder-station-%s.pt" % str(s)
    # Load data
    data_path = "../data/count_by_hour_with_header.csv"
    data_scaled = reader.get_scaled_mrt_data(data_path, [s], datetime_features=True, holidays=HOLIDAYS)
    train_data, val_data, test_data = reader.produce_seq2seq_data(data_scaled, batch_size, input_seq_len,
                                                                  output_seq_len, time_major=True, y_has_features=True)


    x_train, y_train = train_data[0], train_data[1]
    x_val, y_val, = val_data[0], val_data[1]
    x_test, y_test = test_data[0], test_data[1]

    x_train = torch.from_numpy(x_train).contiguous()
    y_train = torch.from_numpy(y_train).contiguous()

    x_val = torch.from_numpy(x_val).contiguous()
    y_val = torch.from_numpy(y_val).contiguous()

    targets_train = y_train[:, :, :, [0]]
    features_train = y_train[:, :, :, 1:]

    targets_val = y_val[:, :, :, [0]]
    features_val = y_val[:, :, :, 1:]

    targets_test = y_test[:, :, :, [0]]
    features_test = y_test[:, :, :, 1:]

    encoder = EncoderRNN(input_size, hidden_size, n_layers, dropout)
    decoder = DecoderRNN(input_size, hidden_size, output_size, n_layers, dropout)

    if os.path.isfile(encoder_checkpoint):
        print("Loading encoder checkpoint...")
        encoder.load_state_dict(torch.load(encoder_checkpoint))

    if os.path.isfile(decoder_checkpoint):
        print("Loading decoder checkpoint...")
        decoder.load_state_dict(torch.load(decoder_checkpoint))

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    if use_cuda:
        encoder.cuda()
        decoder.cuda()

    best_val_loss = 1000
    for epoch in range(n_epochs):
        n_batches = x_train.shape[0]
        train_loss = 0
        for i in range(n_batches):
            input_batches = x_train[i]
            target_batches = targets_train[i]
            decoder_feature_batches = features_train[i]
            train_loss = train(input_batches, target_batches, decoder_feature_batches,
                         encoder, decoder, encoder_optimizer, decoder_optimizer, use_cuda=use_cuda)

        # n_val_batches = x_val.shape[0]
        # val_loss_total = 0
        # for i in range(n_val_batches):
        #     input_batches = x_val[i]
        #     target_batches = targets_val[i]
        #     decoder_feature_batches = features_val[i]
        #     _, loss = evaluate(input_batches, target_batches, decoder_feature_batches, encoder, decoder, use_cuda)
        #     val_loss_total += loss
        #
        # val_loss_mean = val_loss_total / n_val_batches

        _, val_loss_mean = evaluate_batches(x_val, targets_val, features_val, encoder, decoder, use_cuda)

        print("epoch %i/%i" % (epoch+1, n_epochs))
        print("training loss is %f, validation loss is %f" % (train_loss, val_loss_mean))

        if val_loss_mean < best_val_loss:
            print("Saving the model...")
            best_val_loss = val_loss_mean
            torch.save(encoder.state_dict(), encoder_checkpoint)
            torch.save(decoder.state_dict(), decoder_checkpoint)

    predictions, loss = evaluate_batches(x_val, targets_val, features_val, encoder, decoder, use_cuda)
    print("test mse is %f" % loss)
    # plt.plot(predictions.numpy().flatten())
    # plt.plot(targets_val.numpy().flatten())
    # plt.show()



