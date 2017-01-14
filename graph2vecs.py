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
import run_detectors
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import visualisation

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=1.8)]


# construct input-output pairs
class Params:
    def __init__(self, batch_size, embedding_size, neg_samples, skip_window, num_steps, logging_interval):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.neg_samples = neg_samples
        self.skip_window = skip_window
        self.num_steps = num_steps
        self.logging_interval = logging_interval  # the number of batches between loss logs


class Graph2Vecs():
    def __init__(self, vocab_size, unigrams, params):
        # Set the parameters
        self.params = params
        self.vocab_size = vocab_size
        self.batch_size = params.batch_size
        self.embedding_size = params.embedding_size  # Dimension of the embedding vector.
        self.num_samples = self.batch_size * params.neg_samples
        self.epochs_to_train = 1
        self.initial_learning_rate = 0.2
        self.global_step = tf.Variable(0, name="global_step")
        self.n_words = 0  # progress counter
        # self.words_to_train = float(words_per_epoch * self.epochs_to_train)
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
            0.0001, 1.0 - float(n_words) / self.params.num_steps)
        return lr

    # Define the computational graph
    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / self.embedding_size
        self.emb = tf.Variable(
            tf.random_uniform(
                [self.vocab_size, self.embedding_size], -init_width, init_width),
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
            range_max=self.vocab_size,
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


def main(outpath, walks, unigrams, vocab_size, params):
    # initialise the graph
    graph = tf.Graph()
    # run the tensorflow session
    with tf.Session(graph=graph) as session:
        # Define the training data
        model = Graph2Vecs(vocab_size, unigrams, params)

        # initialize all variables in parallel
        tf.global_variables_initializer().run()
        _ = [print(v) for v in tf.global_variables()]

        s = datetime.datetime.now()
        print("Initialized")
        # define batch generator
        batch_gen = generate_batch(params.skip_window, walks, params.batch_size)
        average_loss = 0
        n_words = 0
        for step in xrange(params.num_steps):
            s_batch = datetime.datetime.now()
            batch_inputs, batch_labels = batch_gen.next()
            lr = model.update_lr(n_words)
            feed_dict = {model.lr: lr, model.examples: batch_inputs, model.labels: batch_labels}
            _, loss_val = session.run([model.train, model.loss], feed_dict=feed_dict)
            average_loss += loss_val
            n_words += 1
            if step % params.logging_interval == 0:
                if step > 0:
                    average_loss /= params.logging_interval
                # The average loss is an estimate of the loss over the last 2000 batches.
                runtime = datetime.datetime.now() - s_batch
                print("Average loss at step ", step, ": ", average_loss, 'learning rate is', lr, 'ran in', runtime)
                s_batch = datetime.datetime.now()
                average_loss = 0
        # final_embeddings = normalized_embeddings.eval()
        final_embedding = model.emb.eval()
        np.savetxt(outpath, final_embedding)
        # saver.save(session, 'tf_out/test.ckpt')
        # ckpt = tf.train.get_checkpoint_state('tf_out')
        # saver.restore(session, ckpt.model_checkpoint_path)
        # np.savetxt('resources/test/tf_test2.csv', emb.eval())
        print('ran in {0} s'.format(datetime.datetime.now() - s))
        return final_embedding


def get_vocab_size(adj_path, bipartite):
    """
    Get the number of vertices in the graph (equivalent to the NLP vocab size)
    :param adj_path: the path to the sparse CSR adjacency matrix
    :param bipartite: True if the graph is bipartite
    :return: an integer vocab_size
    """
    adj = utils.read_pickle(adj_path)
    vocab_size = adj.shape[0]
    if bipartite:
        vocab_size += adj.shape[1]
    return vocab_size


def twitter_age_scenario():
    # Read the data
    walks = pd.read_csv('resources/test/node2vec/walks_1.0_1.0.csv', header=None).values
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    vocab_size = get_vocab_size(x_path, bipartite=True)
    # define the noise distribution
    _, unigrams = np.unique(walks, return_counts=True)
    params = Params()
    main('resources/test/tf.emd', walks, unigrams, vocab_size, params)


def karate_results(embedding):
    deepwalk_path = 'local_resources/zachary_karate/size8_walks1_len10.emd'

    y_path = 'local_resources/zachary_karate/y.p'
    x_path = 'local_resources/zachary_karate/X.p'

    target = utils.read_target(y_path)

    if embedding.shape[1] == 2:
        visualisation.plot_embedding(embedding, labels)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['embedding'], ['logistic']]

    # x_deepwalk = utils.read_embedding(deepwalk_path, target)
    # all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)
    X = [normalize(embedding, axis=0), normalize(x, axis=0)]
    # names = ['embedding']
    # X = embedding
    n_reps = 10
    train_size = 4  # the number of labelled points to use
    results = []
    for exp in zip(X, names):
        tmp = run_detectors.run_experiments(exp[0], y, exp[1], classifiers, n_reps, train_size)
        results.append(tmp)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/karate/tf_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/karate/tf_micro_pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = 'results/karate/tf_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/karate/tf_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def karate_scenario():
    walks = pd.read_csv('local_resources/zachary_karate/walks1_len10_p1_q1.csv', header=None).values
    x_path = 'local_resources/zachary_karate/X.p'
    vocab_size = get_vocab_size(x_path, bipartite=False)
    print('vocab of size: ', vocab_size)
    # define the noise distribution
    _, unigrams = np.unique(walks, return_counts=True)
    params = Params(batch_size=4, embedding_size=2, neg_samples=5, skip_window=4, num_steps=3000, logging_interval=100)
    embedding = main('local_resources/zachary_karate/tf.emd', walks, unigrams, vocab_size, params)
    karate_results(embedding)


if __name__ == '__main__':
    s = datetime.datetime.now()
    karate_scenario()
    print(datetime.datetime.now() - s)
