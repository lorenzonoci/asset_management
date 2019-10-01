import tensorflow as tf
from pre_proc import create_batches
import numpy as np
import tensorflow.contrib.layers as layers


class Autoencoder:
    '''
    Implementation of Autoencoder with two hidden layers. We do not actually use it in the project
    '''
    def __init__(self, input_size, hidden_size1, hidden_size2, window_size, activation=tf.nn.relu):

        self.input_state = tf.placeholder(tf.float32, shape=[None, window_size, input_size], name="input_state")
        initializer = tf.contrib.layers.xavier_initializer()

        embed_dim = 20
        #embed_dim = 10

        w1 = tf.Variable(initializer([input_size*window_size, hidden_size1*embed_dim]), dtype=tf.float32)
        w2 = tf.Variable(initializer([hidden_size1*embed_dim, hidden_size2*embed_dim]), dtype=tf.float32)
        w3 = tf.Variable(initializer([hidden_size2*embed_dim, hidden_size1*embed_dim]), dtype=tf.float32)
        w4 = tf.Variable(initializer([hidden_size1*embed_dim, input_size*window_size]), dtype=tf.float32)

        b1 = tf.Variable(tf.zeros(hidden_size1*embed_dim))
        b2 = tf.Variable(tf.zeros(hidden_size2*embed_dim))
        b3 = tf.Variable(tf.zeros(hidden_size1*embed_dim))
        b4 = tf.Variable(tf.zeros(input_size*window_size))

        reshaped_input = tf.reshape(self.input_state, shape=[-1, window_size*input_size])
        hid_layer1 = activation(tf.matmul(reshaped_input, w1) + b1)
        self.embedding = activation(tf.matmul(hid_layer1, w2) + b2)
        self.hidden_state = tf.reshape(self.embedding, shape=[-1, embed_dim, hidden_size2], name="embedding")
        hid_layer3 = activation(tf.matmul(self.embedding, w3) + b3)
        self.output_state = tf.matmul(hid_layer3, w4) + b4
        #reshaped_output = tf.reshape(self.output_state, shape=[-1, window_size, input_size])

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.square(reshaped_input - self.output_state))
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00002)
            self.optimizer = self.optimizer.minimize(self.loss)

    @staticmethod
    def train_autoencoder(data, hidden_size1=14, hidden_size2=5, batch_size=64, num_epochs=40, window_size=1):
        input_size = data.shape[-1]
        batches = create_batches(data, batch_size=batch_size)

        with tf.Graph().as_default():
            autoenc = Autoencoder(input_size=input_size,
                                  hidden_size1=hidden_size1,
                                  hidden_size2=hidden_size2,
                                  window_size=window_size)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(num_epochs):
                    batch_loss = 0
                    for batch in batches:
                        feed_dict = {
                            autoenc.input_state: batch
                        }
                        loss, opt, out = sess.run(
                            [autoenc.loss, autoenc.optimizer, autoenc.output_state],
                            feed_dict=feed_dict
                        )
                        batch_loss += loss
                    print("Average loss in epoch {}: {}".format(epoch+1, batch_loss/len(batches)))
                saver = tf.train.Saver()
                saver.save(sess, "autoencoder/model", global_step=num_epochs)
            return autoenc


    @staticmethod
    def get_embedding(data, net_config):
        with tf.Graph().as_default():
            saver = tf.train.import_meta_graph('autoencoder/model-{}.meta'.format(net_config.num_epochs))
            with tf.Session() as sess:
                saver.restore(sess, tf.train.latest_checkpoint("autoencoder/"))
                emb = tf.get_default_graph().get_tensor_by_name(name="embedding:0")
                states = sess.run(emb,
                                  feed_dict={tf.get_default_graph().get_tensor_by_name(name="input_state:0"): data})
        return states
