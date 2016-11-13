"""
Efficient random walk generation
"""
import numpy as np
from numpy.random import randint
from datetime import datetime
import utils
import pandas as pd
from scipy.sparse import csr_matrix

class Graph:
    """
    A binary graph
    """

    def __init__(self, adj):
        self.adj = adj.tocsr()
        self.deg = np.array(adj.sum(axis=1), dtype=int).squeeze()
        self.n_vertices = self.deg.shape[0]
        self.edges = np.zeros(shape=(self.n_vertices, max(self.deg)), dtype=int)

    def build_edge_array(self):
        """
        construct an array that represents edges in a manner that can be vector sampled
        :return:
        """
        for row_idx in range(self.n_vertices):
            z = self.adj[row_idx, :].nonzero()[1]
            self.edges[row_idx, 0:len(z)] = z

    def get_idx(self, x):
        return randint(x)

    def generate_walks(self, num_walks, walk_length):
        """
        generate random walks
        :param num_walks the number of random walks per vertex
        :param walk_length the length of each walk
        :return:
        """
        degs = np.tile(self.deg, num_walks)
        edges = np.tile(self.edges, (num_walks, 1))
        initial_vertices = np.arange(self.n_vertices)
        walks = np.zeros(shape=(self.n_vertices * num_walks, walk_length), dtype=int)
        walks[:, 0] = np.tile(initial_vertices, num_walks)
        for walk_idx in range(walk_length - 1):
            # get the degree of the vertices we're starting from
            current_vertices = walks[:, walk_idx]
            current_degrees = degs[current_vertices]
            # get the indices of the next vertices. This is the random bit
            col_idx = np.array(map(self.get_idx, current_degrees))
            walks[:, walk_idx + 1] = edges[current_vertices, col_idx]
        return walks


def read_data(threshold):
    """
    reads the features and target variables
    :return:
    """
    x_path = 'local_resources/blogcatalog/X.p'
    y_path = 'resources/test/y.p'
    X = utils.read_pickle(x_path)
    X1, cols = utils.remove_sparse_features(X, threshold=threshold)
    print X1.shape
    return X1


if __name__ == '__main__':
    print 'reading data'
    x = read_data(0)
    s = datetime.now()
    #x = csr_matrix(np.array([[0, 1], [1, 0]]))
    g = Graph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    print datetime.now() - s, ' s'
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv('local_resources/blogcatalog/walks1.csv', index=False, header=None)
