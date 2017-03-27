from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import utils

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

__author__ = 'benchamberlain'

FLAGS = None


class tf_model:
    """
    wrapper for a tf model so it fits in a scikit-learn shaped hole
    """

    def __init__(self, vars, train_step, session, batch_size, training_iters):
        self.vars = vars
        self.train_step = train_step
        self.session = session
        self.batch_size = batch_size
        self.n_iter = training_iters

    def fit(self, data):
        """
        fit the model
        :param data is an MLDataset object
        """
        # need to one hot encode
        data.train.target = tf.one_hot(data.train.target, depth=10, dtype=tf.float32)
        # Train
        for _ in range(self.n_iter):
            # take a random sample of the data for batch gradient descent
            x_slice, y_slice = data.train.next_batch(batch_size=self.batch_size)
            self.session.run(train_step, feed_dict={self.vars['x']: x_slice, self.vars['y_']: y_slice})

    def predict(self, data):
        """
        predict unseen test values
        :param data is an MLDataset object
        """
        test_features = np.array(data.test.features.todense())
        preds = self.session.run(tf.argmax(self.vars['y'], 1), feed_dict={self.vars['x']: test_features})
        return preds


def read_data(threshold):
    """
    Read the Twitter user ages test data
    :param threshold: the minimum number of edges each
    :return:
    """
    x_path = 'resources/test/X.p'
    y_path = 'resources/test/y.p'
    X = utils.read_pickle(x_path)
    features, cols = utils.remove_sparse_features(X, threshold=threshold)
    # features = np.array(X.todense())
    targets = utils.read_pickle(y_path)
    target = np.array(targets['cat'])
    unique, counts = np.unique(target, return_counts=True)
    total = float(sum(counts))
    norm_counts = counts / total
    print(np.asarray((unique, norm_counts)).T)
    return features, target


def build_model(input_units, output_units):
    """
    construct a tensor flow graph
    :param input_units: the number of input units
    :param output_units: the number of output units
    :return:
    """
    x = tf.placeholder(tf.float32, [None, input_units])
    W = tf.Variable(tf.zeros([input_units, output_units]))
    b = tf.Variable(tf.zeros([output_units]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # This returns a function that performs a single step of gradient descent through backprop
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    vars = {'x': x, 'W': W, 'b': b, 'y': y, 'y_': y_}

    return train_step, vars



def accuracy(y, pred):
    return sum(y == pred) / float(len(y))


def main():
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    features, target = read_data(threshold=5)
    n_data, n_features = features.shape

    # Create the model
    # Placeholders allow the DAG to be constructed in python. The second parameter is the shape
    # NOT SURE WHY THESE ARE FLOATS WHEN THE DATA IS INTEGER - MAYBE TF ONLY WORKS WITH FLOATS??
    x = tf.placeholder(tf.float32, [None, n_features])
    # in tensorflow params are usually read into variables
    W = tf.Variable(tf.zeros([n_features, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # This returns a function that performs a single step of gradient descent through backprop
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    # tf.initialize_all_variables().run()
    for _ in range(2):
        #     # batch_xs, batch_ys = mnist.train.next_batch(100)
        idx = np.random.choice(3000, 100)
        y_slice = target.eval()[idx, :]
        x_slice = np.array(features[idx, :].todense())
        sess.run(train_step, feed_dict={x: x_slice, y_: y_slice})

    # Test trained model
    y_test = target[3000:, :].eval()
    x_test = np.array(features[3000:, :].todense())
    print(tf.argmax(y_test, 1).eval())
    print(sess.run(tf.argmax(y, 1), feed_dict={x: x_test}))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x_test,
                                        y_: y_test}))


if __name__ == '__main__':
    X, y = read_data(threshold=5)
    train_step, vars = build_model(X.shape[1], 10)
    # using an interactive session allows us to interleave building and running elements of the graph
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf_model = tf_model(vars, train_step, sess, batch_size=100, training_iters=1000)
    y_pred = utils.run_cv_pred(X, y, n_folds=2, model=tf_model)
    utils.get_metrics(y, y_pred)
