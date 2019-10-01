import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

scaler = StandardScaler()


def get_data(windows_size=None, scale=True):
    '''
    get data from file and optionally scale and compute windows
    :param windows_size:
    :param scale:
    :return:
    '''
    #df = get_df_close().pct_change().dropna()
    df = get_df_close()
    data = df.values
    if scale:
        data = scaler.fit_transform(df)
    if windows_size is not None:
        data = compute_windows(data, windows_size)
    return data


def get_df_close():
    path = "data/markets_new.csv"
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop("Date", axis=1, inplace=True)
    return df


def get_labels(data, window_size, output_window):
    '''
    Compute labels for lstm applying a rolling windows to the input data
    :param data: input data
    :param window_size: window of past data
    :param output_window: prediction time period window
    :return: numpy array of shape [n_labels, output_window, n_stocks]
    '''
    labels = []
    for i in range(len(data)-window_size-output_window+1):
        label = data[i+window_size: i+window_size+output_window]
        labels.append(label)
    labels = np.stack(labels, axis=0)
    return labels


def compute_windows(data, windows_size):
    '''
    Apply rolling window to the input data
    :param data: input data
    :param windows_size: width of the window
    :return: numpy array of shape [n_data, window_size, n_stocks]
    '''
    windows = []
    for i in range(len(data) - windows_size + 1):
        win = data[i:i+windows_size]
        windows.append(win)
    windows = np.stack(windows, axis=0)
    return windows


def get_train_test(data, windows_size, output_window):
    '''
    Get train and test data from input data, also compute rolling window to get
    the complete train and test datasets
    :param data: input data
    :param windows_size:
    :param output_window:
    :return: 4 arrays: training data, test data, train labels and test labels
    '''

    train_data, test_data = train_test_split(data, shuffle=False, test_size=0.15)
    train_labels = get_labels(train_data, window_size=windows_size, output_window=output_window)
    test_labels = get_labels(test_data, window_size=windows_size, output_window=output_window)
    # removing last few elements because we do not have a prediction for it
    train_data = train_data[:-output_window]
    test_data = test_data[:-output_window]
    train_data = compute_windows(train_data, windows_size=windows_size)
    test_data = compute_windows(test_data, windows_size=windows_size)
    return train_data, test_data, train_labels, test_labels


def concat(data, labels):
    dataset = np.concatenate([data[:, 0, :], data[-1, 1:, :], labels[-1, :, :]])
    return dataset


def create_batches(input, batch_size):
    """Divides `input` and `target` in batches of size `batch_size`.
    The parameters `input` and `target` should have equal length.

    Parameters
    ----------
    input
        The input words
    batch_size
        The predefined batch size
    """
    n_batches = len(input) // batch_size
    batched_input = []
    n = 0
    for n in range(n_batches):
        batched_input.append(input[n*batch_size: (n+1)*batch_size])
    if len(input[(n+1)*batch_size:]) > 0:
        batched_input.append(input[(n+1)*batch_size:])
    return batched_input
