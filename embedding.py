"""
Take a bipartite network and generate an embedding vector for each node using the gensim word2vec package
TODO: adapt this to work for bipartite graphs
"""

from utils import *
from gensim.models import Word2Vec

graph = read_pickle('resources/X.p')

# normalise the graph
graph =


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


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=64, window=10, min_count=0, sg=1, workers=4,
                     iter=1)
    model.save_word2vec_format('../emb/test.emd')


if __name__ == '__main__':
    pass
