import tensorflow as tf
import numpy as np


# A general ANN
class Network:
    # Initialize weight, bias
    # The Size is to define input layer, hidden layer, output layer dimension
    def __init__(self, size):
        self.size = size
        self.weight = [tf.Variable(np.random.randn((m, n))) for m, n in zip(size[1:], size[:-1])]
        self.bias = [tf.Variable(np.random.rand((m, 1)) for m in size[1:])]
        self.init = tf.global_variables_initializer()

    def fit(self, epoch, learning_rate, x, y):
        # Placeholder for training is just one vector because it is SGD
        x_place = tf.compat.v1.placeholder(tf.float64, shape=(x.shape[2], 1))
        y_place = tf.compat.v1.placeholder(tf.float64, shape=(self.weight[-1], 1))
        a = self.forward(x_place)

        # Back-propagation will be solved by tensorflow
        diff = y_place - a
        loss = tf.reduce_mean(diff ** 2)
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        self.init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(epoch):
                x_feed = x.iloc[i, :].values.reshape(-1, 1)  # Tranpose to column
                feed_dict = {x: x_feed, y: y[i].reshape(-1, 1)}
                print("Epoch: {}/{}".format(i, epoch))
                print("Error: {}".format(sess.run(loss, feed_dict=feed_dict)))
                sess.run(train, feed_dict=feed_dict)

    def forward(self, x):
        # Define step before proceeding to epoch
        a = x  # Create a copy
        for weight, bias in zip(self.weight, self.bias):
            z = tf.matmul(weight, a) + bias
            a = tf.sigmoid(z)
        return a
    tf.matmul
