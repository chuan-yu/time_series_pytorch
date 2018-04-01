import numpy as np
import pandas as pd
from datetime import datetime


def generate_sin_signal(x, noisy=False):
    y = np.sin(x)
    if noisy:
        noise = np.random.rand(len(x)) * 0.2
        y = y + noise

    y = np.reshape(y, (y.shape[0], 1))
    return y


def _read_raw_mrt_data(data_path=None):
    '''Read data from a csv file.

    :param data_path: the path of data file.
    :return: pandas data frame of size [time_steps, num_stations].

    '''
    data = pd.read_csv(data_path, index_col=0, dtype=np.float32)
    data.reset_index(drop=True, inplace=True)
    data = pd.DataFrame.transpose(data)
    return data


def _read_raw_bus_data(stops, data_path=None):
    '''Read data from a csv file.
    :param stops: a list of strings
    :param data_path: the path of data file.
    :return: pandas data frame of size [time_steps, num_stops].
    '''

    df = pd.read_csv(data_path)
    df.drop_duplicates(['Boarding_stop_stn', 'Ride_start_datetime'], keep='first', inplace=True)
    df = df.loc[df['Boarding_stop_stn'].isin(stops)]
    df = df.pivot(index='Ride_start_datetime', columns='Boarding_stop_stn', values='count')

    return df


def _scale(data):
    '''Rescale the data by column to the range between 0 and 1.

    For each column, X = (X - X.min()) / (X.max() - X.min())

    :param data: numpy 2-d array
    :return: numpy 2-d array
    '''

    data_mins = data.min(axis=0)
    data_mins = data_mins.reshape((-1, data_mins.shape[0]))
    data = np.subtract(data, data_mins)
    minmax_range = data.max(axis=0) - data.min(axis=0)
    minmax_range = minmax_range.reshape((-1, minmax_range.shape[0]))
    data = np.divide(data, minmax_range)
    return data

def get_scaled_mrt_data(data_path=None, stations_codes=None, datetime_features=False, holidays=None):
    ''' Get scaled mrt data

    :param stations_codes: a list of selected station codes. If None, return data for all stations
    :param data_path: file path of the data
    :return: numpy array of size [time_steps, num_stations]
    '''
    # read raw data as pandas Dataframe
    raw_data = _read_raw_mrt_data(data_path)
    if stations_codes is not None:
        raw_data = raw_data[stations_codes]

    if datetime_features:
        raw_data.index = pd.to_datetime(raw_data.index)
        index = raw_data.index + pd.DateOffset(hours=1)
        hours = np.array(index.hour, dtype=np.float32)
        day_week = np.array(index.dayofweek, dtype=np.float32)
        raw_data['hour'] = hours
        raw_data['day_of_week'] = day_week
        # raw_data['dt'] = index.date
        # raw_data['is_workday'] = raw_data['day_of_week'] // 5 == 0
        if holidays is not None:
            holidays = [datetime.strptime(h, '%Y-%m-%d') for h in holidays]
            for h in holidays:
                date = h.date()
                raw_data.ix[(raw_data['dt']==date), 'is_workday'] = False

        # raw_data['is_workday'] = raw_data['is_workday'].astype(int)
        # raw_data.drop(['dt'], axis=1, inplace=True)

    data = raw_data.as_matrix(columns=None)
    data = _scale(data)

    return data


def get_scaled_bus_data(stops, data_path=None, resample_freq=None, datetime_features=False):

    df_raw = _read_raw_bus_data(stops, data_path)
    df_raw.index = pd.to_datetime(df_raw.index)

    if resample_freq is not None:
        df_raw = df_raw.resample(resample_freq, how="sum").fillna(0)

    if datetime_features:
        index = df_raw.index
        minutes = np.array(index.minute)
        hours = np.array(index.hour)
        day_week = np.array(index.dayofweek)
        df_raw['day_of_week'] = day_week
        df_raw['hour'] = hours
        df_raw['minute'] = minutes
        df_raw[['day_of_week', 'hour', 'minute']] = df_raw[['day_of_week', 'hour', 'minute']].shift(-1)


    df_raw.drop(df_raw.index[-1], inplace=True)
    data = df_raw.as_matrix()
    data = _scale(data)

    return data

def mrt_simple_lstm_data(data, batch_size, truncated_backpro_len,
                         train_ratio=0.6, val_ratio=0.2):
    '''Produce the training, validation and test set

    :param data: time series data of shape [time_steps, feature_len]
    :param batch_size: training data batch size
    :param truncated_backpro_len: the number of time steps with which backpropogation through time is done. Also equal to
    the time window size.
    :param num_prediction_steps: the number of time steps in the future to be predicted.
    :param train_ratio: the ratio of training samples to the total data samples. The value is a float between 0 and 1.
    Default is 0.6.
    :param val_ratio: the ration of training samples to the total data samples. The value is a float between 0 and 1.
    Default is 0.2.
    :param data_path: the file path to the data file
    :return:
            (x_train, y_train), (x_val, y_val), (x_test, y_test)
            x_train: [num_train_batches, batch_size, truncated_backpro_len, feature_len],
            y_train: [num_train_batches, batch_size, feature_len],
            x_val: [num_val_samples, truncated_backpro_len, feature_len]
            y_val: [num_val_samples, feature_len]
            x_test: [num_test_samples, truncated_backpro_len, feature_len]
            y_test: [num_test_samples, feature_dim]
    '''

    total_data_len = data.shape[0]
    num_features = data.shape[1]

    seq_len = truncated_backpro_len + 1
    num_time_windows = total_data_len - seq_len
    time_windows = []

    for i in range(num_time_windows):
        time_windows.append(data[i:i + seq_len, :])

    time_windows = np.array(time_windows)

    num_train = round(num_time_windows * train_ratio)
    train = time_windows[0:num_train, :, :]

    num_val = round(num_time_windows * val_ratio)
    val = time_windows[num_train:num_train + num_val, :, :]

    test = time_windows[num_train + num_val:, :, :]

    num_train_batches = num_train // batch_size
    train = train[0:num_train_batches *  batch_size, :, :]

    train = np.reshape(train, (-1, batch_size, train.shape[1], train.shape[2]))
    x_train = train[:, :, :-1, :]
    y_train = train[:, :, -1, :]

    num_val_batches = num_val // batch_size
    val = val[0:num_val_batches * batch_size, :, :]

    x_val = val[:, :-1, :]
    y_val = val[:, -1, :]

    x_test = test[:, :-1, :]
    y_test = test[:, -1, :]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def produce_seq2seq_data(data, batch_size, input_seq_len, output_seq_len,
                         time_major=False, y_has_features=False,
                         train_ratio=0.6, val_ratio=0.2):
    ''' Produce seq2seq data

    :param data: time series data of shape [time_steps, feature_len]
    :param batch_size: training data batch size
    :param input_seq_len: the input sequence length for every training sample
    :param output_seq_len: the output sequence length for every training sample
    :param time_major: whether outputs are time-majored.
    :param train_ratio: the ratio of training samples to the total data samples. The value is a float between 0 and 1.
    Default is 0.6.
    :param val_ratio: val_ratio: the ration of training samples to the total data samples. The value is a float between 0 and 1.
    Default is 0.2.
    :return:
            (x_train, y_train), (x_val, y_val), (x_test, y_test)
            if time_major:
                x_train: [num_train_batches, input_seq_len, batch_size, feature_len],
                y_train: [num_train_batches, output_seq_len, batch_size, feature_len],
                x_val: [num_val_batches, input_seq_len, 1, feature_len]
                y_val: [num_val_batches,output_seq_len, 1, feature_len]
                x_test: [num_test_batches, input_seq_len, 1, feature_len]
                y_test: [num_test_batches, output_seq_len, 1, feature_len]
            else:
                x_train: [num_train_batches, batch_size, input_seq_len, feature_len],
                y_train: [num_train_batches, batch_size, output_seq_len, feature_len],
                x_val: [num_val_batches, 1, input_seq_len, feature_len]
                y_val: [num_val_batches, 1, output_seq_len, feature_len]
                x_test: [num_test_batches, 1, input_seq_len, feature_len]
                y_test: [num_test_batches, 1, output_seq_len, feature_len]

    '''

    total_time_steps = data.shape[0]



    num_train = round(total_time_steps * train_ratio)
    num_val = round(total_time_steps * val_ratio)

    train_raw = data[0:num_train]
    val_raw = data[num_train-input_seq_len-1:num_train + num_val]
    test_raw = data[num_train + num_val - input_seq_len - 1:]


    total_seq_len = input_seq_len + output_seq_len

    train_window = _convert_to_windows(train_raw, total_seq_len)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(train_window.shape[0]))
    train_window = train_window[shuffle_indices]

    val_window = _convert_to_windows(val_raw, total_seq_len, False, output_seq_len)
    test_window = _convert_to_windows(test_raw, total_seq_len, False, output_seq_len)

    num_train_batches = train_window.shape[0] // batch_size
    train_window = train_window[0:num_train_batches * batch_size]

    train = np.reshape(train_window, (num_train_batches, batch_size, -1, train_window.shape[2]))
    val = np.expand_dims(val_window, axis=1)
    test = np.expand_dims(test_window, axis=1)

    if time_major:

        # it time_major, the 2nd aix is the sequence length
        train = np.swapaxes(train, 1, 2)
        val = np.swapaxes(val, 1, 2)
        test = np.swapaxes(test, 1, 2)

        x_train = train[:, :-output_seq_len, :, :]
        x_val = val[:, :-output_seq_len, :, :]
        x_test = test[:, :-output_seq_len, :, :]

        if not y_has_features:
            y_train = train[:, -output_seq_len:, :, [0]]
            y_val = val[:, -output_seq_len:, :, [0]]
            y_test = test[:, -output_seq_len:, :, [0]]
        else:
            y_train = train[:, -output_seq_len:, :, :]
            y_val = val[:, -output_seq_len:, :, :]
            y_test = test[:, -output_seq_len:, :, :]

        # test_time_features = test[:, -output_seq_len:, :, 1:]

    else:
        # if not time_major, the 3rd axis is the sequence length

        x_train = train[:, :, :-output_seq_len, :]
        x_val = val[:, :, :-output_seq_len, :]
        x_test = test[:, :, :-output_seq_len, :]

        if not y_has_features:
            y_train = train[:, :, -output_seq_len:, [0]]
            y_val = val[:, :, -output_seq_len:, [0]]
            y_test = test[:, :, -output_seq_len:, [0]]

        else:
            y_train = train[:, :, -output_seq_len:, :]
            y_val = val[:, :, -output_seq_len:, :]
            y_test = test[:, :, -output_seq_len:, :]

        # test_time_features = test[:, :, -output_seq_len:, 1:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def _convert_to_windows(data, total_seq_len, train=True, output_seq_len=1):
    '''Convert time series data to time windows of seq_len
    Example:
        data = [0, 1, 2, 3, 4, 5, 6]
        total_seq_len = 5
        output_seq_len = 2
        if train:
            return [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        if not train:
            return [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]]
            when train is False, time windows are taken by shifting output_seq_len positions to the right

    :param data: input data of shape [time_steps, feature_len]
    :param total_seq_len: sequence length
    :param output_seq_len: output sequence length
    :param train: whether the output is for training
    :return: numpy array of shape [num_time_windows, total_seq_len, feature_len]
    '''
    if train:
        forward_steps = 1
    else:
        forward_steps = output_seq_len

    num_time_windows = (data.shape[0] - total_seq_len) // forward_steps + 1
    time_windows = np.ndarray((num_time_windows, total_seq_len, data.shape[1]), dtype=np.float32)

    for i in range(num_time_windows):
        time_windows[i] = data[i * forward_steps:i * forward_steps + total_seq_len, :]

    return time_windows


def get_pretrain_data(data, batch_size, time_steps):
    x = data[:-1]
    y = data[1:, 0]

    n = x.shape[0] // (batch_size * time_steps)
    x = x[0:n * batch_size * time_steps]
    y = y[0:n * batch_size * time_steps]

    x = x.reshape((-1, time_steps, batch_size, x.shape[1]))
    y = y.reshape((-1, time_steps, batch_size, 1))

    return x, y


