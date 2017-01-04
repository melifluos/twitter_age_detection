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

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


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


data_index = 0
row_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch_tf(batch_size, num_skips, skip_window):
    """
    This function takes a contiguous block of data of length 2*skip_window + 1 staring at the data_index.
    It uses the middle value as the input value and samples num_skips labels randomly from the remaining values. It then
    shifts the block of data and the input value up one position and repeats until batch_size input-output pairs have
    been generated.
    :param batch_size: The number of input-output pairs
    :param num_skips: The number of target labels to sample for each input value
    :param skip_window: 2*skip_window + 1 is the total length of data considered for input-output pairs.
    :return: numpy array of shape batch_size of input values, numpy array of shape (batch_size, 1) of labels
    """
    # define data_index to have global scope so that other functions etc. can use it
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # a deque is like a list but with efficient reads / writes to both ends
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        # this is data_index++ but it loops back to 0 when data_index = len(data)
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):  # // is the divide and floor operator
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):  # sample num_skips different labels for each input word from this buffer
            while target in targets_to_avoid:  # keep sampling until we select a different value
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])  # this appends a value to the end and pops off the first value
        data_index = (data_index + 1) % len(data)
    return batch, labels


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


# def generate_batch(skip_window, data):
#     global data_index
#     sentence = data[data_index, :]
#     data_index = (data_index + 1) % data.shape[0]
#     return process_sentence(sentence, skip_window)


def generate_batch(skip_window, data):
    row_index = 0
    while True:
        sentence = data[row_index, :]
        for pos, word in enumerate(sentence):
            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - skip_window)
            # enumerate takes a second arg, which sets the starting point, this makes pos and pos2 line up
            for pos2, word2 in enumerate(sentence[start: pos + skip_window + 1], start):
                if pos2 != pos:
                    yield [word], [[word2]]
        row_index = (row_index + 1) % data.shape[0]


# filename = maybe_download('text8.zip', 31344016)

# words = read_data(filename)
# print('Data size', len(words))
#
# # Step 2: Build the dictionary and replace rare words with UNK token.
# vocabulary_size = 50000

# data, count, dictionary, reverse_dictionary = build_dataset(words)
# del words  # Hint to reduce memory.
# print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#     print(batch[i], reverse_dictionary[batch[i]],
#           '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

walks = pd.read_csv('resources/test/node2vec/walks_1.0_1.0.csv', header=None).values
x_path = 'resources/test/balanced7_100_thresh_X.p'
y_path = 'resources/test/balanced7_100_thresh_y.p'
x, y = utils.read_data(x_path, y_path, threshold=1)
n_data, n_features = x.shape
vocabulary_size = n_data + n_features


def degree_sort(x):
    data_degs = x.sum(axis=1)
    feature_degs = x.sum(axis=0)
    all_degs = np.concatenate((data_degs, feature_degs.T), axis=0)
    sorted_idx = np.argsort(-np.array(all_degs), axis=None)
    dic = {x: idx for idx, x in enumerate(sorted_idx)}
    reverse_idx = np.argsort(sorted_idx, axis=None)
    return dic, reverse_idx


dic, reverse_idx = degree_sort(x)


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


ordered_walks = vec_translate(walks, dic)

del walks

batch_size = 1  # PROBABLY WAY TOO BIG FOR A BATCH
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 10  # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.
num_sampled = 5  # Number of negative examples to sample for the batch - NEED TO CHECK EXACTLY WHAT THIS IS DOING
num_steps = n_data * 10

batch_gen = generate_batch(skip_window, ordered_walks)


# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
# valid_size = 16  # Random set of words to evaluate similarity on.
# valid_window = 100  # Only pick dev samples in the head of the distribution.
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

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
    optimizer = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    # valid_embeddings = tf.nn.embedding_lookup(
    #     normalized_embeddings, valid_dataset)
    # similarity = tf.matmul(
    #     valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
# num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    s = datetime.datetime.now()
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = batch_gen.next()
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            # if step % 10000 == 0:
            #     sim = similarity.eval()
            #     for i in xrange(valid_size):
            #         valid_word = reverse_dictionary[valid_examples[i]]
            #         top_k = 8  # number of nearest neighbors
            #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            #         log_str = "Nearest to %s:" % valid_word
            #         for k in xrange(top_k):
            #             close_word = reverse_dictionary[nearest[k]]
            #             log_str = "%s %s," % (log_str, close_word)
            #         print(log_str)
    final_embeddings = normalized_embeddings.eval()
    # put back in the same order as the labels
    final_embeddings = final_embeddings[reverse_idx]
    np.savetxt('resources/test/tf_test2.csv', final_embeddings)
    print
    'ran in {0} s'.format(datetime.datetime.now() - s)


# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

# try:
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#     plot_only = 500
#     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#     labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#     plot_with_labels(low_dim_embs, labels)
#
# except ImportError:
#     print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


# The negative sampling routing from gensim L256
# if model.negative:
#     # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
#     word_indices = [predict_word.index]
#     while len(word_indices) < model.negative + 1:
#         w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
#         if w != predict_word.index:
#             word_indices.append(w)
#     l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
#     fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
#     gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
#     if learn_hidden:
#         model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
#     neu1e += dot(gb, l2b)  # save error
