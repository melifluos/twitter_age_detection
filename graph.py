"""
Efficient random walk generation
"""
import numpy as np
from numpy.random import randint
from datetime import datetime
import utils


class Graph:
    """
    A binary graph
    """

    def __init__(self, adj):
        self.adj = adj.tocsr()
        self.deg = np.array(adj.sum(axis=1)).squeeze()
        self.n_vertices = self.deg.shape[0]
        self.edges = np.zeros(shape=(self.n_vertices, max(self.deg)), dtype=int)

    def build_edge_array(self):
        """
        construct an array that represents edges in a manner that can be vector sampled
        :return:
        """
        for row_idx in range(self.n_vertices):
            z = self.adj[0, :].nonzero()[1]
            self.edges[row_idx, 0:len(z)] = z

    def get_idx(self, x):
        return randint(low=0, high=x)

    def generate_walks(self, num_walks, walk_length):
        """
        generate random walks
        :param num_walks the number of random walks per vertex
        :param walk_length the length of each walk
        :return:
        """
        walks = np.zeros(shape=(self.n_vertices, walk_length), dtype=int)
        for walk_idx in range(walks.shape[1] - 1):
            # get the degree of the vertices we're starting from
            current_degrees = self.deg[walks[:, walk_idx]]
            # get the indices of the next vertices
            col_idx = np.array(map(self.get_idx, current_degrees))
            walks[:, walk_idx + 1] = self.edges[walks[:, walk_idx], col_idx]
        return walks


def read_data(threshold):
    """
    reads the features and target variables
    :return:
    """
    x_path = 'resources/test/X.p'
    y_path = 'resources/test/y.p'
    X = utils.read_pickle(x_path)
    X1, cols = utils.remove_sparse_features(X, threshold=threshold)
    print X1.shape
    return X1


if __name__ == '__main__':
    print 'reading data'
    x = read_data(0)
    s = datetime.now()
    g = Graph(x)
    g.build_edge_array()
    walks = g.generate_walks(1, 10)
    print datetime.now() - s, ' s'
