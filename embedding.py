"""
Take a bipartite network and generate an embedding vector for each node using the gensim word2vec package
TODO: adapt this to work for bipartite graphs
"""

from gensim.models import Word2Vec
import networkx as nx
import node2vec
import age_detector
import utils
import gensim
import csv
import datetime


# graph = read_pickle('resources/X.p')


def random_walk():
    """

    :return:
    """


def build_sentences():
    """
    A sentence is a list of node indices that exist within a single context
    :return:
    """
    pass


class WalkLines:
    """
    A class to pass to word2vec that lets lines be streamed from disk. Useful as largish graphs overflow memory
    """

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with gensim.utils.smart_open(self.path) as fin:
            reader = csv.reader(fin)
            for line in reader:
                yield line


def learn_embeddings(walks, size, outpath):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]

    model = Word2Vec(walks, size=size, window=10, min_count=0, sg=1, workers=4,
                     iter=1)
    model.save_word2vec_format(outpath)


def learn_embeddings_file(inpath, size, outpath):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = WalkLines(inpath)

    model = Word2Vec(walks, size=size, window=10, min_count=0, sg=1, workers=4,
                     iter=1)
    model.save_word2vec_format(outpath)


def main(size, num_walks, walk_len, paths):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = nx.read_edgelist(paths[0], nodetype=int, create_using=nx.DiGraph())
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1
    nx_G = nx_G.to_undirected()
    G = node2vec.Graph(nx_G, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=num_walks, walk_length=walk_len)
    learn_embeddings(walks, size, paths[1])


def main1(size, num_walks, walk_len, paths):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print 'creating networkx graph object'
    # SOMETHING IS HAPPENING HERE SO THAT THE DEGREE OF MY MATRIX AND THE DEGREE OF THIS GRAPH ARE DIFFERENT
    nx_G = nx.read_edgelist(paths[0], nodetype=int, create_using=nx.DiGraph())
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1
    nx_G = nx_G.to_undirected()
    print 'creating node2vec graph object'
    G = node2vec.Graph(nx_G, False, 1, 1)
    print 'pre-processing transition probabilites'
    G.preprocess_transition_probs()
    print 'writing random walks to file'
    G.output_walks(num_walks=num_walks, walk_length=walk_len, path=paths[2])
    print 'learning embeddings'
    learn_embeddings_file(paths[2], size, paths[1])


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

    s = datetime.datetime.now()
    inpaths = ['resources/test/blogcatalog/blogcatalog.edgelist', 'resources/test/flickr/flickr.edgelist',
               'resources/test/youtube/youtube.edgelist']
    outpaths = ['resources/test/blogcatalog/blogcatalog128.emd', 'resources/test/flickr/flickr128.emd',
                'resources/test/youtube/youtube128.emd']
    walkpaths = ['resources/test/blogcatalog/walks.csv', 'resources/test/flickr/walks.csv',
                'resources/test/youtube/walks.csv']
    for paths in zip(inpaths, outpaths, walkpaths):
        main1(size=128, num_walks=10, walk_len=80, paths=paths)
    print 'ran in {0} s' .format(datetime.datetime.now() - s)
    #import pandas as pd
    # edge_list = pd.read_csv('resources/test/test.edgelist', names=['fan_idx', 'star_idx'], sep=' ', dtype=int)
    # X = utils.edge_list_to_sparse_mat(edge_list)
    # #X = read_data(threshold=0)
    # paths = ['resources/test/test.edgelist', ' ', 'resources/test/walks.csv']
    # s = datetime.datetime.now()
    # main1(64, 1, 10, paths, X)
    # # learn_embeddings_file('resources/walks.csv', 64, 'resources/walks.emd')
    # print 'ran in {0} s'.format(datetime.datetime.now() - s)
