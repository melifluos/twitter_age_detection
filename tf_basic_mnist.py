from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from utils import *
from sklearn.preprocessing import OneHotEncoder

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from sklearn.cross_validation import KFold, StratifiedKFold

__author__ = 'benchamberlain'

FLAGS = None


class Train:
    def __init__(self):
        self.features = None
        self.target = None


class Test:
    def __init__(self):
        self.features = None
        self.target = None


class MLdata(object):
    """
    supervised ml data object
    """

    def __init__(self):
        self.train = Train
        self.test = Test


def read_data(threshold):
    """
    Read the Twitter user ages test data
    :param threshold: the minimum number of edges each
    :return:
    """
    x_path = 'resources/X.p'
    y_path = 'resources/y.p'
    X = read_pickle(x_path)
    features, cols = remove_sparse_features(X, threshold=threshold)
    # features = np.array(X.todense())
    targets = read_pickle(y_path)
    target = np.array(targets['cat'])
    unique, counts = np.unique(target, return_counts=True)
    total = float(sum(counts))
    norm_counts = counts / total
    print(np.asarray((unique, norm_counts)).T)
    return features, target


def run_cv_pred(X, y, n_folds=3):
    """
    Run n-fold cross validation returning a prediction for every row of X
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :param clf: The
    :param n_folds:
    :return:
    """
    # Construct a kfolds object
    kf = StratifiedKFold(y, n_folds=n_folds)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        data = MLdata()
        data.train.features, data.test.features = X[train_index], X[test_index]
        data.train.target, data.test.target = y[train_index], y[test_index]

        # Initialize a classifier with key word arguments
        preds = run_classifier(data, batch_size=100)
        y_pred[test_index] = preds
    return y_pred


def run_classifier(data, batch_size=100, n_iter=2000):
    """
    Run the tensor flow multinomial logistic regression classifier
    :param data: an MLdata object
    :return:
    """
    n_data, n_features = data.train.features.shape
    # need to onehotencode
    data.train.target = tf.one_hot(data.train.target, depth=10, dtype=tf.float32)

    x = tf.placeholder(tf.float32, [None, n_features])
    W = tf.Variable(tf.zeros([n_features, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # This returns a function that performs a single step of gradient descent through backprop
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    # Train
    # tf.initialize_all_variables().run()
    for _ in range(n_iter):
        # take a random sample of the data for batch gradient descent
        idx = np.random.choice(n_data, batch_size)
        y_slice = data.train.target.eval()[idx, :]
        x_slice = np.array(data.train.features[idx, :].todense())
        sess.run(train_step, feed_dict={x: x_slice, y_: y_slice})

    test_features = np.array(data.test.features.todense())
    preds = sess.run(tf.argmax(y, 1), feed_dict={x: test_features})
    return preds


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

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # This returns a function that performs a single step of gradient descent through backprop
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
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
    features, target = read_data(threshold=5)
    y_pred = run_cv_pred(features, target, n_folds=2)
    print(y_pred[0:20])
    print(accuracy(target, y_pred))
