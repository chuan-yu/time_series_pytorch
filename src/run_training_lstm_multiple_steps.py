import os.path
from models.lstm import LSTM
from train.train_multiple_steps import train, evaluate, predict_batches, get_mse
import reader
import torch
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

use_cuda = True


input_size = 4
output_size = 10
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
    data_path = "../data/count_by_hour_with_header.csv"
    checkpoint_file = "../checkpoints/simple_lstm_multiple_steps/lstm-station-%s.pt" % str(s)
    data_scaled = reader.get_scaled_mrt_data(data_path, [s], datetime_features=True, holidays=HOLIDAYS)

    n = data_scaled.shape[0]

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


    lstm = LSTM(input_size, hidden_size, output_size, n_layers, dropout)

    if os.path.isfile(checkpoint_file):
        print("Loading checkpoint...")
        lstm.load_state_dict(torch.load(checkpoint_file))

    if use_cuda:
        lstm.cuda()

    # optimizer = optim.Adam(lstm.parameters(), lr=lr)
    #
    # best_val_loss = 1000
    # train_loss = 0
    # for epoch in range(n_epochs):
    #     n_batches = x_train.shape[0]
    #     for i in range(n_batches):
    #         lstm.hidden = None
    #         input_batches = x_train[i]
    #         target_batches = targets_train[i]
    #         train_loss = train(input_batches, target_batches, lstm, optimizer, use_cuda)
    #
    #     epoch_train_loss = evaluate(x_train, targets_train, lstm, use_cuda)
    #     epoch_val_loss = evaluate(x_val, targets_val, lstm, use_cuda)
    #
    #     print("epoch %i/%i" % (epoch + 1, n_epochs))
    #     print("traing loss is %f, validation loss is %f" % (epoch_train_loss, epoch_val_loss))
    #
    #     if epoch_val_loss < best_val_loss:
    #         print("Saving the model...")
    #         best_val_loss = epoch_val_loss
    #         torch.save(lstm.state_dict(), checkpoint_file)


    predictions, test_mse= predict_batches(x_val, targets_val, lstm, use_cuda=use_cuda)
    print("test mse is %f" % test_mse)
    plt.plot(predictions.numpy().flatten())
    plt.plot(targets_val.numpy().flatten())
    plt.show()