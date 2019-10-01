import tensorflow as tf
import tensorflow.contrib.layers as layers


class LSTM:
    '''
    LSTM based neural network tensor flow implementation
    '''
    def __init__(self, input_size, output_size, window_size, window_output, rnn_hidden, learning_rate):
        # graph input/output size
        self.input_size = input_size
        self.output_size = output_size

        self.inputs = tf.placeholder(tf.float32, (None, window_size, self.input_size), name="lstm_input")  # (batch, time, in)
        self.outputs = tf.placeholder(tf.float32, (None, window_output, self.output_size), name="lstm_output") # (batch, time, out)
        self.dropout_prob = tf.placeholder_with_default(1.0, shape=(), name="is_training")
        cell = tf.keras.layers.LSTMCell(rnn_hidden)
        rnn = tf.keras.layers.RNN(cell, return_sequences=True)
        rnn_outputs = rnn(self.inputs)

        rnn_outputs = tf.reshape(rnn_outputs, shape=[-1, window_size*rnn_hidden])
        rnn_outputs = layers.dropout(rnn_outputs, keep_prob=self.dropout_prob)

        self.predicted_outputs = layers.linear(rnn_outputs,
                                               num_outputs=self.output_size*window_output,
                                               activation_fn=None
                                               )
        self.predicted_outputs = tf.reshape(self.predicted_outputs, shape=[-1, window_output, output_size], name="predicted_output")

        error = tf.squared_difference(
            self.predicted_outputs, self.outputs)
        self.loss = tf.reduce_mean(error)

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

        self.saver = tf.train.Saver()

    @staticmethod
    def train_lstm(sess, model, data_batches, labels_batches, test_data, test_labels, n_epochs, dropout_prob, initialize=True):
        if initialize:
            sess.run(tf.global_variables_initializer())
        for e in range(n_epochs):
            ep_loss = 0.0
            for i in range(len(data_batches)):
                feed_dict = {
                    model.inputs: data_batches[i],
                    model.outputs: labels_batches[i],
                    model.dropout_prob: dropout_prob
                }
                l, _, o = sess.run([model.loss, model.train_fn, model.predicted_outputs],
                                   feed_dict=feed_dict)
                ep_loss += l

            feed_dict_eval = {
                model.inputs: test_data,
                model.outputs: test_labels,
                model.dropout_prob: 1.0
            }
            for feed_dict in [feed_dict_eval]:
                eval_loss, preds = sess.run([model.loss, model.predicted_outputs], feed_dict=feed_dict)
                print("Eval loss in epoch {}: {}".format(e + 1, eval_loss))
            print("Average train loss in epoch {}: {}".format(e + 1, ep_loss / len(data_batches)))
            print()
        return model
