import numpy as np
import tensorflow as tf

class ConvNet:

    def __init__(self, session, num_classes = 4, learning_rate=0.001, net_name = "default"):
        self.session = session
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.net_name = net_name

    def build_network(self, state_shape = [100, 100], batch_size = 32):
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(tf.float32, shape=[None, state_shape[0], state_shape[1], 4])

            conv1 = tf.layers.conv2d(self.X, 32, kernel_size=8, strides=4, padding="same", activation=tf.nn.relu)

            conv2 = tf.layers.conv2d(conv1, 64, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu)

            conv3 = tf.layers.conv2d(conv2, 64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            conv3_flat = tf.reshape(conv3, [-1, conv3.shape[1] * conv3.shape[2] * conv3.shape[3]])

            net = tf.layers.dense(conv3_flat, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.num_classes)

            self.inference = net
            self.predict = tf.argmax(self.inference, 1)

            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
            self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

    def predict_val(self, X):
        return self.session.run(self.inference, feed_dict = {self.X : X})

    def train_net(self, X, Y):
        return self.session.run(self.optimizer, feed_dict = {self.X : X, self.Y : Y})

    def set_weights(self, src_name):
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.net_name)

        for i in range(len(src_vars)):
            assign_op = dest_vars[i].assign(src_vars[i])
            self.session.run(assign_op)

