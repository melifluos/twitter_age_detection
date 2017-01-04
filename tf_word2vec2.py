# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import datetime


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def process_sentence(sentence, skip_window):
    """
    For a whole sentence of usually 80 words, generate all input-output pairs that exist within skip-window
    For 80 words this will generate 770 pairs
    :param sentence:
    :param skip_window:
    :return:
    """
    batch = []
    labels = []
    for pos, word in enumerate(sentence):
        # now go over all words from the window, predicting each one in turn
        start = max(0, pos - skip_window)
        # enumerate takes a second arg, which sets the starting point, this makes pos and pos2 line up
        for pos2, word2 in enumerate(sentence[start: pos + skip_window + 1], start):
            if pos2 != pos:
                batch.append(word)
                labels.append([word2])
    return batch, labels


def generate_batch(skip_window, data):
    global data_index
    sentence = data[data_index, :]
    data_index = (data_index + 1) % data.shape[0]
    return process_sentence(sentence, skip_window)


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


def degree_sort(x):
    data_degs = x.sum(axis=1)
    feature_degs = x.sum(axis=0)
    all_degs = np.concatenate((data_degs, feature_degs.T), axis=0)
    sorted_idx = np.argsort(-np.array(all_degs), axis=None)
    dic = {x: idx for idx, x in enumerate(sorted_idx)}
    reverse_idx = np.argsort(sorted_idx, axis=None)
    return dic, reverse_idx


def forward(examples, labels, opts):
    """Build the graph for the forward pass."""

    # Declare all variables we need.
    # Embedding: [vocab_size, emb_dim]
    init_width = 0.5 / opts.emb_dim
    emb = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size, opts.emb_dim], -init_width, init_width),
        name="emb")
    self._emb = emb

    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    sm_w_t = tf.Variable(
        tf.zeros([opts.vocab_size, opts.emb_dim]),
        name="sm_w_t")

    # Softmax bias: [emb_dim].
    sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

    # Global step: scalar, i.e., shape [].
    self.global_step = tf.Variable(0, name="global_step")

    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [opts.batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=opts.num_samples,
        unique=True,
        range_max=opts.vocab_size,
        distortion=0.75,
        unigrams=opts.vocab_counts.tolist()))

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
    sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
    sampled_logits = tf.matmul(example_emb,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits


def nce_loss(true_logits, sampled_logits, batch_size):
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


# walks = pd.read_csv('resources/test/node2vec/walks_1.0_1.0.csv', header=None).values
# x_path = 'resources/test/balanced7_100_thresh_X.p'
# y_path = 'resources/test/balanced7_100_thresh_y.p'
# x, y = utils.read_data(x_path, y_path, threshold=1)
# n_data, n_features = x.shape
# data_index = 0
# row_index = 0
#
# dic, reverse_idx = degree_sort(x)
#
# ordered_walks = vec_translate(walks, dic)

# del walks

batch_size = 770  # PROBABLY WAY TOO BIG FOR A BATCH
batch_size = 2
skip_window = 5  # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.
num_sampled = 385  # Number of negative examples to sample for the batch - NEED TO CHECK EXACTLY WHAT THIS IS DOING
num_sampled = 1
# num_steps = n_data * 5
num_steps = 3
embedding_size = 2  # Dimension of the embedding vector.
# vocabulary_size = n_data + n_features
vocabulary_size = 3

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        # nce_weights = tf.Variable(np.zeros(shape=(vocabulary_size, embedding_size), dtype=np.float32))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    s = datetime.datetime.now()
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        # batch_inputs, batch_labels = generate_batch(skip_window, ordered_walks)
        # feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        feed_dict = {train_inputs: [1, 1], train_labels: [[0], [0]]}
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        print('embeddings at step', str(step))
        print(embeddings.eval())
        print('embed at step', str(step))
        print(embed.eval(feed_dict))
        print('nce weights at step', str(step))
        print(nce_weights.eval())
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

    print('embeddings at step', str(step))
    print(embeddings.eval())
    final_embeddings = normalized_embeddings.eval()
    # put back in the same order as the labels
    # final_embeddings = final_embeddings[reverse_idx]
    np.savetxt('resources/test/tf_test4.csv', embeddings.eval())
    print
    'ran in {0} s'.format(datetime.datetime.now() - s)
