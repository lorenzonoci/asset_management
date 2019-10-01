from pre_proc import get_data, create_batches, concat
import numpy as np
from config_parse import net_config
from lstm import LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from optimize_portfolio import show_results, compute_performance, backtesting_optim_portfolio
from pre_proc import scaler, get_train_test
from tensorflow.core.protobuf import rewriter_config_pb2

config_proto = tf.ConfigProto()

off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off



def pipeline(output_window):
    '''
    Apply training of lstm and portfolio optimization
    :param output_window: time windows length of LSTM predictions
    :return:
    '''
    data_autoen = get_data(scale=True)
    train_data, test_data, train_labels, test_labels = get_train_test(data_autoen,
                                                                      windows_size=net_config.windows_size,
                                                                      output_window=output_window)
    #data = get_data(windows_size=net_config.windows_size)
    batches = create_batches(train_data, batch_size=net_config.batch_size)
    labels_batches = create_batches(train_labels, batch_size=net_config.batch_size)

    with tf.Graph().as_default():
        lstm_model = LSTM(input_size=data_autoen.shape[-1],
                          output_size=data_autoen.shape[-1],
                          rnn_hidden=net_config.rnn_hidden,
                          window_size=net_config.windows_size,
                          window_output=output_window,
                          learning_rate=net_config.learning_rate,
                          )
        with tf.Session(config=config_proto) as sess:
            if net_config.lstm_from_file:
                lstm_model.saver.restore(sess, tf.train.latest_checkpoint('lstm-price/'))
            else:
                LSTM.train_lstm(sess, lstm_model, batches, labels_batches, test_data, test_labels,
                                net_config.lstm_epochs, net_config.dropout_prob)
                lstm_model.saver.save(sess, "lstm-price/model", global_step=net_config.lstm_epochs)
            '''
            feed_dict = {
                lstm_model.inputs: test_data[0:1, :, :],
                lstm_model.dropout_prob: 1.0
            }
            
            preds = sess.run(lstm_model.predicted_outputs, feed_dict=feed_dict)
            n_stocks = data_autoen.shape[-1]
            fig, axes = plt.subplots(int(n_stocks / 2), 2)
            i = 0
            for n in range(int(n_stocks / 2)):
                for j in range(2):
                    axes[n, j].plot(preds[0, :, i], label='predicted')
                    axes[n, j].plot(test_labels[0, :, i], label='true')
                    i += 1
            plt.legend()
            plt.show()
            '''

            df_close = pd.DataFrame(test_data[:, -1, :])
            df_close = pd.DataFrame(scaler.inverse_transform(df_close))
            df_close = df_close.pct_change().dropna()

            train_dataset = concat(train_data, train_labels)
            all_weights, pnls = backtesting_optim_portfolio(
                df_close, test_data, test_labels, train_dataset, lstm_model, sess, net_config.risk_aversion, output_window, net_config)
            print(all_weights.shape)

            performances = compute_performance(pnls)
            print(performances)
            return np.array(performances), all_weights, pnls


output_windows = [net_config.output_window]
all_performances = []
all_weights = []
all_pnls = []
for window in output_windows:
    performances, weights, pnls = pipeline(output_window=window)
    all_performances.append(performances)
    all_weights.append(weights)
    all_pnls.append(pnls)

for i in range(len(output_windows)):
    np.save("all_weights_{}".format(i), all_weights[i])
np.save("all_performances", np.stack(all_performances, axis=0))
for i in range(len(output_windows)):
    np.save("all_pnls_{}".format(i), all_pnls[i])


