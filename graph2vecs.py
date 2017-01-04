"""
This uses negative sampling and the skip-gram model to embed vertices of graphs based on short random walks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Read the data
walks = pd.read_csv('resources/test/node2vec/walks_1.0_1.0.csv', header=None).values
x_path = 'resources/test/balanced7_100_thresh_X.p'
y_path = 'resources/test/balanced7_100_thresh_y.p'
x, y = utils.read_data(x_path, y_path, threshold=1)
n_data, n_features = x.shape
vocab_size = n_data + n_features
# define the noise distribution
_, unigrams = np.unique(walks, return_counts=True)
words_per_epoch = n_data * 770

# Set the parameters
batch_size = 16
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 10  # How many words to consider left and right.
num_samples = batch_size * 5  # Number of negative examples to sample for the batch
num_steps = 1000000
epochs_to_train = 1
initial_learning_rate = 1.0
global_step = tf.Variable(0, name="global_step")
n_words = 0


# construct input-output pairs


# define the optimisation
def optimize(loss):
    """Build the graph to optimize the loss function."""

    # Optimizer nodes.
    # Linear learning rate decay.
    words_to_train = float(words_per_epoch * epochs_to_train)
    lr = initial_learning_rate * tf.maximum(
        0.0001, 1.0 - tf.cast(n_words, tf.float32) / words_to_train)
    if n_words % 1600 == 0:
        print('processed', n_words, 'pairs. Current learning rate is ', lr)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss, gate_gradients=optimizer.GATE_NONE)
    return train


# produce batch of data
def generate_batch(skip_window, data, batch_size):
    """
    A generator that produces the next batch of examples and labels
    :param window: The largest distance between an example and a label
    :param data:  the random walks
    :return:
    """
    row_index = 0
    examples = []
    labels = []
    while True:
        sentence = data[row_index, :]
        for pos, word in enumerate(sentence):
            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - skip_window)
            # enumerate takes a second arg, which sets the starting point, this makes pos and pos2 line up
            for pos2, word2 in enumerate(sentence[start: pos + skip_window + 1], start):
                if pos2 != pos:
                    examples.append(word)
                    labels.append([word2])
                    if len(examples) == batch_size:
                        yield examples, labels
                        examples = []
                        labels = []
        row_index = (row_index + 1) % data.shape[0]


# Define the computational graph
def forward(examples, labels):
    """Build the graph for the forward pass."""
    # Embedding: [vocab_size, emb_dim]
    init_width = 0.5 / embedding_size
    emb = tf.Variable(
        tf.random_uniform(
            [vocab_size, embedding_size], -init_width, init_width),
        name="emb")

    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    sm_w_t = tf.Variable(
        tf.zeros([vocab_size, embedding_size]),
        name="sm_w_t")

    # Softmax bias: [emb_dim].
    sm_b = tf.Variable(tf.zeros([vocab_size]), name="sm_b")

    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=num_samples,
        unique=True,
        range_max=vocab_size,
        distortion=0.75,
        unigrams=unigrams.tolist()))

    # Embeddings for examples: [batch_size, emb_dim]
    example_emb = tf.nn.embedding_lookup(emb, examples)

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)

    # Weights for sampled ids: [num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [num_samples])
    sampled_logits = tf.matmul(example_emb,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits, emb


# define the loss function
def nce_loss(true_logits, sampled_logits):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        true_logits, tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        sampled_logits, tf.zeros_like(sampled_logits))

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / batch_size
    return nce_loss_tensor


# put it all together
def build_graph(examples, labels):
    true_logits, sampled_logits, emb = forward(examples, labels)
    loss = nce_loss(true_logits, sampled_logits)
    train = optimize(loss)
    return train, loss, emb


# initialise the graph
graph = tf.Graph()
# run the tensorflow session
with tf.Session(graph=graph) as session:
    # Define the training data
    examples = tf.placeholder(tf.int32, shape=[batch_size])
    labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # build the graph
    train, loss, emb = build_graph(examples, labels)
    # Add opp to save variables
    saver = tf.train.Saver()
    # initialize all variables in parallel
    tf.global_variables_initializer().run()
    s = datetime.datetime.now()
    print("Initialized")
    # define batch generator
    batch_gen = generate_batch(skip_window, walks, batch_size)
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = batch_gen.next()
        feed_dict = {examples: batch_inputs, labels: batch_labels}
        _, loss_val = session.run([train, loss], feed_dict=feed_dict)
        average_loss += loss_val
        n_words += batch_size
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
    # final_embeddings = normalized_embeddings.eval()

    np.savetxt('resources/test/tf_test5.csv', emb.eval())
    # saver.save(session, 'tf_out/test.ckpt')
    # ckpt = tf.train.get_checkpoint_state('tf_out')
    # saver.restore(session, ckpt.model_checkpoint_path)
    # np.savetxt('resources/test/tf_test2.csv', emb.eval())
    print('ran in {0} s'.format(datetime.datetime.now() - s))
