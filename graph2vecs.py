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
words_per_epoch = n_data * 10 * 1490
skip_window = 10
batch_size = 160
num_steps = 10000

#TODO run this on the karate club and debug it properly

karate_path = 'local_resources/zachary_karate/walks1_len10_p1_q1.csv'

# construct input-output pairs

class Graph2Vecs():
    def __init__(self, vocab_size, words_per_epoch, batch_size, unigrams):
        # Set the parameters
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.neg_samples_per_example = 4
        self.num_samples = self.batch_size * self.neg_samples_per_example
        self.epochs_to_train = 1
        self.initial_learning_rate = 0.2
        self.global_step = tf.Variable(0, name="global_step")
        self.n_words = 0
        self.words_to_train = float(words_per_epoch * self.epochs_to_train)
        self.examples = tf.placeholder(tf.int32, shape=[self.batch_size], name='examples')
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='labels')
        self.lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.unigrams = unigrams
        # Add opp to save variables
        self.saver = tf.train.Saver()

        true_logits, sampled_logits = self.forward(self.examples, self.labels)
        self.loss = self.nce_loss(true_logits, sampled_logits)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train = optimizer.minimize(self.loss, global_step=self.global_step,
                                        gate_gradients=optimizer.GATE_NONE)

    # update the learning rate
    def update_lr(self, n_words):
        """
        linear decay of learning rate. Update the optimizer after each batch
        :param n_words: the number of output-context pairs seen so far
        :return:
        """
        lr = self.initial_learning_rate * max(
            0.0001, 1.0 - float(n_words) / self.words_to_train)
        return lr

    # Define the computational graph
    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / self.embedding_size
        self.emb = tf.Variable(
            tf.random_uniform(
                [vocab_size, self.embedding_size], -init_width, init_width),
            name="emb")

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([self.vocab_size, self.embedding_size]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [self.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,
            range_max=vocab_size,
            distortion=0.75,
            unigrams=self.unigrams.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(self.emb, examples)

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
        sampled_b_vec = tf.reshape(sampled_b, [self.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    # define the loss function
    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            true_logits, tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            sampled_logits, tf.zeros_like(sampled_logits))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size
        return nce_loss_tensor

        # # put it all together
        # def build_graph(self, examples, labels):
        #     """
        #     build the forward graph, the loss function and the optimizer
        #     :param examples: training inputs
        #     :param labels: training labels
        #     :return:
        #     """
        #     true_logits, sampled_logits = self.forward(examples, labels)
        #     loss = self.nce_loss(true_logits, sampled_logits)
        #     # tf.contrib.deprecated.scalar_summary("NCE loss", loss)
        #     self.loss = loss
        #     self.train = self.optimize(loss)


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


# initialise the graph
graph = tf.Graph()
# run the tensorflow session
with tf.Session(graph=graph) as session:
    # Define the training data
    model = Graph2Vecs(vocab_size, words_per_epoch, batch_size, unigrams)

    # initialize all variables in parallel
    tf.global_variables_initializer().run()
    _ = [print(v) for v in tf.global_variables()]

    s = datetime.datetime.now()
    print("Initialized")
    # define batch generator
    batch_gen = generate_batch(skip_window, walks, batch_size)
    average_loss = 0
    n_words = 0
    for step in xrange(num_steps):
        s_batch = datetime.datetime.now()
        batch_inputs, batch_labels = batch_gen.next()
        lr = model.update_lr(n_words)
        feed_dict = {model.lr: lr, model.examples: batch_inputs, model.labels: batch_labels}
        _, loss_val = session.run([model.train, model.loss], feed_dict=feed_dict)
        average_loss += loss_val
        n_words += len(batch_inputs)
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            runtime = datetime.datetime.now() - s_batch
            print("Average loss at step ", step, ": ", average_loss, 'learning rate is', lr, 'ran in', s_batch)
            s_batch = datetime.datetime.now()
            average_loss = 0
    # final_embeddings = normalized_embeddings.eval()

    np.savetxt('resources/test/tf_test6.csv', model.emb.eval())
    # saver.save(session, 'tf_out/test.ckpt')
    # ckpt = tf.train.get_checkpoint_state('tf_out')
    # saver.restore(session, ckpt.model_checkpoint_path)
    # np.savetxt('resources/test/tf_test2.csv', emb.eval())
    print('ran in {0} s'.format(datetime.datetime.now() - s))
