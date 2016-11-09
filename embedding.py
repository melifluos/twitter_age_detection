"""
Take a bipartite network and generate an embedding vector for each node using the gensim word2vec package
TODO: adapt this to work for bipartite graphs
"""

from gensim.models import Word2Vec
import networkx as nx
import node2vec
import age_detector
from utils import *


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


def learn_embeddings(walks, size, outpath):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=size, window=10, min_count=0, sg=1, workers=4,
                     iter=1)
    model.save_word2vec_format(outpath)


def main(size, paths):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = nx.read_edgelist(paths[0], nodetype=int, create_using=nx.DiGraph())
    for edge in nx_G.edges():
       nx_G[edge[0]][edge[1]]['weight'] = 1
    nx_G = nx_G.to_undirected()
    G = node2vec.Graph(nx_G, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(10, 80)
    learn_embeddings(walks, size, paths[1])


if __name__ == '__main__':
    inpaths = ['resources/test/blogcatalog/blogcatalog.edgelist', 'resources/test/flickr/flickr.edgelist',
               'resources/test/youtube/youtube.edgelist']
    outpaths = ['resources/test/blogcatalog/blogcatalog128.emd', 'resources/test/flickr/flickr128.emd',
                'resources/test/youtube/youtube128.emd']
    for paths in zip(inpaths, outpaths):
        main(128, paths)