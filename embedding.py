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
                     iter=5)
    model.save_word2vec_format(outpath)


def learn_embeddings_file(inpath, size, outpath):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = WalkLines(inpath)

    model = Word2Vec(walks, size=size, window=10, min_count=0, sg=1, workers=4,
                     iter=5)
    model.save_word2vec_format(outpath)


def output_walks(walks, path):
    """
    write the walks to a csv file
    :param walks: A list of walks. Each walk is a list of ints
    :return:
    """
    with open(path, 'wb') as f:
        writer = csv.writer(f)
        for walk in walks:
            writer.writerow(walk)


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
    output_walks(walks, paths[2])
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
    G = node2vec.Graph(nx_G, False, 0.25, 0.25)
    print 'pre-processing transition probabilites'
    G.preprocess_transition_probs()
    G.output_walks(num_walks=num_walks, walk_length=walk_len, path=paths[2])
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


def scenario_pq_grid():
    """
    Generate age embeddings for every p,q combination used in the node2vec paper writing them to file
    :return:
    """
    print 'creating networkx graph object'
    inpath = 'resources/test/balanced7_100_thresh.edgelist'
    # SOMETHING IS HAPPENING HERE SO THAT THE DEGREE OF MY MATRIX AND THE DEGREE OF THIS GRAPH ARE DIFFERENT
    nx_G = nx.read_edgelist(inpath, nodetype=int, create_using=nx.DiGraph())
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1.0
    nx_G = nx_G.to_undirected()
    print 'creating node2vec graph object'
    walk_stub = 'resources/test/node2vec/walks_'
    emd_stub = 'resources/test/node2vec/'
    for p in [0.25, 0.5, 1, 2, 4]:
        for q in [0.25, 0.5, 1, 2, 4]:
            print 'running p={0}, q={1}'.format(str(p), str(q))
            walk_path = walk_stub + str(p) + '_' + str(q) + '.csv'
            emd_path = emd_stub + str(p) + '_' + str(q) + '.emd'
            G = node2vec.Graph(nx_G, False, p, q)
            print 'pre-processing transition probabilites'
            G.preprocess_transition_probs()
            G.output_walks(num_walks=10, walk_length=80, path=walk_path)
            learn_embeddings_file(walk_path, size=128, outpath=emd_path)


def scenario_generate_public_embeddings(size=128):
    inpaths = ['local_resources/blogcatalog/blogcatalog.edgelist', 'local_resources/flickr/flickr.edgelist',
               'local_resources/youtube/youtube.edgelist']
    outpaths = ['local_resources/blogcatalog/blogcatalog128.emd', 'local_resources/flickr/flickr128.emd',
                'local_resources/youtube/youtube128.emd']
    walkpaths = ['local_resources/blogcatalog/walks.csv', 'local_resources/flickr/walks.csv',
                 'local_resources/youtube/walks.csv']
    for paths in zip(inpaths, outpaths, walkpaths):
        main(size=size, num_walks=10, walk_len=80, paths=paths)


def scenario_generate_blogcatalog_embedding(size=128):
    paths = ['local_resources/blogcatalog/blogcatalog.edgelist',
             'local_resources/blogcatalog/blogcatalog_p025_q025_d128.emd',
             'local_resources/blogcatalog/p025_q025_d128_walks.csv']
    main1(size=size, num_walks=10, walk_len=80, paths=paths)


def scenario_generate_small_age_detection_embedding():
    import pandas as pd
    edge_list = pd.read_csv('resources/test/test.edgelist', names=['fan_idx', 'star_idx'], sep=' ', dtype=int)
    X = utils.edge_list_to_sparse_mat(edge_list)
    # X = read_data(threshold=0)
    paths = ['resources/test/test.edgelist', 'resources/test/test1281.emd', 'resources/test/walks1.csv']
    s = datetime.datetime.now()
    main(128, 10, 80, paths)
    # learn_embeddings_file('resources/walks.csv', 64, 'resources/walks.emd')
    print 'ran in {0} s'.format(datetime.datetime.now() - s)


if __name__ == '__main__':
    s = datetime.datetime.now()
    scenario_pq_grid()
    # import pandas as pd
    # walks = pd.read_csv('local_resources/blogcatalog/p025_q025_d128_walks.csv', header=None, index_col=0, skiprows=1)
    # print walks.head()
    # learn_embeddings(walks, 128, 'local_resources/blogcatalog/p025_q025_d128_walks.emd')
    print 'ran in {0} s'.format(datetime.datetime.now() - s)
