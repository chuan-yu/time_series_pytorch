from models.seq2seq import EncoderRNN, DecoderRNN
from train.train import train, evaluate
import torch
import reader
from torch import optim
import numpy as np

use_cuda = torch.cuda.is_available()

input_size = 3
output_size = 1
hidden_size = 200
n_layers = 1
lr = 0.0005
batch_size = 32
n_epochs = 1000
input_seq_len = 50
output_seq_len = 10

# Load data
data_path = "../data/count_by_hour_with_header.csv"
data_scaled = reader.get_scaled_mrt_data(data_path, [0], datetime_features=True, holidays=None)

n = data_scaled.shape[0]
x_pretrain, y_pretrain = reader.get_pretrain_data(data_scaled[0:round(0.6 * n)],
                                                  3, input_seq_len)

train_data, val_data, test_data = reader.produce_seq2seq_data(data_scaled, batch_size, input_seq_len,
                                                              output_seq_len, time_major=True, y_has_features=True)


x_train, y_train = train_data[0], train_data[1]
x_val, y_val, = val_data[0], val_data[1]
x_test, y_test = test_data[0], test_data[1]

targets_train = y_train[:, :, :, [0]]
features_train = y_train[:, :, :, 1:]

targets_val = y_val[:, :, :, [0]]
features_val = y_val[:, :, :, 1:]

targets_test = y_test[:, :, :, [0]]
features_test = y_test[:, :, :, 1:]

encoder = EncoderRNN(input_size, hidden_size, n_layers)
decoder = DecoderRNN(input_size, hidden_size, output_size, n_layers)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

if use_cuda:
    encoder.cuda()
    decoder.cuda()

for epoch in range(n_epochs):
    n_batches = x_train.shape[0]
    train_loss = 0
    for i in range(n_batches):
        input_batches = x_train[i]
        target_batches = targets_train[i]
        decoder_feature_batches = features_train[i]
        train_loss = train(input_batches, target_batches, decoder_feature_batches,
                     encoder, decoder, encoder_optimizer, decoder_optimizer)

    n_val_batches = x_val.shape[0]
    val_loss_total = 0
    decoder_outputs = np.zeros_like(targets_val, dtype=np.float32)
    for i in range(n_val_batches):
        input_batches = x_val[i]
        target_batches = targets_val[i]
        decoder_feature_batches = features_val[i]
        output = evaluate(input_batches, target_batches, decoder_feature_batches, encoder, decoder)
        decoder_outputs[i] = output
        # val_loss_total += val_loss

    # val_loss_mean = val_loss_total / n_val_batches
    val_loss_mean = np.mean(np.square(decoder_outputs - targets_val))

    print("epoch %i/%i" % (epoch+1, n_epochs))
    print("training loss is %f, validation loss is %f" % (train_loss, val_loss_mean))

