"""
utility functions for detection of age of Twitter users
TODO:
- Construct two files sorted by fan_idx
1/ fan_idx star_idx
2/ fan_idx cat
- Use these to construct sparse matrix and target values.
"""

from scipy.sparse import lil_matrix
import pandas as pd
import cPickle as pickle
import numpy as np

__author__ = 'benchamberlain'


def edge_list_to_sparse_mat(edge_list):
    """
    Convert an edge list into a sparse matrix
    :param edge_list: The edge list in the form fan_id star_idx
    :return: A Columnar sparse matrix
    """
    # Create matrix representation (adjacency matrix) of edge list
    data_shape = edge_list.max(axis=0)
    print 'building sparse matrix of size {0}'.format(data_shape)
    X = lil_matrix((data_shape['fan_index'] + 1, data_shape['star_index'] + 1), dtype=int)
    X[edge_list['fan_index'].values, edge_list['star_index'].values] = 1
    return X.tocsc()


def balance_classes(input_df, n_cat2=23000, n_cat9=1000):
    """
    balances the input data classes so as not to induce incorrect biases in the output
    :param input_df: the raw input data
    :param n_cat2: The number of cat 2 examples to retain
    :param n_older the minimum number of cat 7, 8 and 9 to keep. In reality it might be a bit more as granpeople are
    split over three classes and so making this exact was more trouble than it was worth
    """
    np.random.seed(10)
    cat2 = input_df[input_df['cat'] == 2]
    if len(cat2) > n_cat2:
        rows = np.random.choice(cat2.index.values, n_cat2, replace=False)
        cat2 = cat2.ix[rows]
    cat9 = input_df[input_df['cat'] == 9]
    if len(cat9) > n_cat9:
        rows = np.random.choice(cat9.index.values, n_cat9, replace=False)
        cat9 = cat9.ix[rows]
    input_df = input_df[~input_df['cat'].isin([2, 9])]
    input_df = pd.concat([input_df, cat2, cat9])
    return input_df


def construct_edge_list(path, include_implicits=False, balance_classes=False):
    """
    Assigns each fan an index and constructs an edge list (fan_idx, star_idx) that is used to make a sparse design
    matrix
    :param path: path to data in the format fan_id,star_id,star_idx,num_followers,cat,weight
    :param include_implicits: Include inferred age classes
    :param balance: the input classes
    :return: A pandas DF where each row is an edge of the graph and columns are [fan_idx, star_idx]
    :return: A pandas DF of target variables
    """
    target_df = pd.read_csv(path)
    # remove any grandparents
    target_df = target_df[target_df['weight'] == 1]

    print target_df['cat'].value_counts()

    if balance_classes:
        # reduce the counts in some classes
        target_df = balance_classes(target_df)
    print 'using category frequencies: '
    print target_df['cat'].value_counts().sort_index() / target_df.shape[0]
    # Set the index of each fan
    target_df['index'] = range(target_df.shape[0])
    target_df = target_df[['index', 'fan_id', 'cat']]
    # target_df.to_csv('help.csv')
    target_df.columns = ['fan_index', 'fan_id', 'cat']
    y = target_df['cat']
    print '{0} target values'.format(len(y))
    y.to_csv('local_resources/target_ages.csv')
    # join the new fan index to the star index
    all_data = fan_star.merge(target_df)
    # Just keep the fan index and the star index
    edge_list = all_data[['fan_index', 'index']]
    edge_list.columns = ['fan_index', 'star_index']
    edge_list.to_csv('local_resources/edge_list.csv', index=False)
    return edge_list, np.array(y)


def remove_duplicate_labelled_fans():
    fans = pd.read_csv('resources/raw_data/labelled_fans.csv')
    fans = fans.drop_duplicates('fan_id')
    fans[['fan_id', 'age']].to_csv('resources/labelled_fans.csv', index=False)


def get_fan_idx():
    """
    Switch the fan_ids for indices - better for anonymity and making sparse matrics
    :return: writes resources/fan_list.csv
    """
    fans = pd.read_csv('resources/labelled_fans.csv')
    # The duplicates are quite error prone so it is possible to drop them all by setting
    # parameter keep=False this might cause problems with unindexed fans later though.
    fans = fans.drop_duplicates('fan_id')
    fans = fans.reset_index()
    fans[['index', 'fan_id']].to_csv('resources/fan_list.csv', index=False)


def pickle_sparse(sparse, path):
    """
    Writes a sparse matrix to disk in the python cPickle format
    :param sparse: A scipy s
    :param path:
    :return:
    """
    with open(path, 'wb') as outfile:
        pickle.dump(sparse, outfile, protocol=2)


def read_pickle(path):
    with open(path, 'rb') as infile:
        return pickle.load(infile)


if __name__ == "__main__":
    get_fan_idx()
