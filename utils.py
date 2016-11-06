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
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold

__author__ = 'benchamberlain'


class MLData:
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def next_batch(self, batch_size):
        """
        sample a batch of data
        """
        n_data, _ = self.features.shape
        idx = np.random.choice(n_data, batch_size)
        target_batch = self.target.eval()[idx, :]
        feature_batch = np.array(self.features[idx, :].todense())
        return feature_batch, target_batch


class MLdataset(object):
    """
    supervised ml data object
    """

    def __init__(self, train, test):
        self.train = train
        self.test = test


def get_metrics(y, pred):
    print 'macro'
    print f1_score(y, pred, average='macro')
    print 'micro'
    print f1_score(y, pred, average='micro')
    print 'all'
    print f1_score(y, pred, average=None)
    # return sum(y == pred) / float(len(y))


def run_cv_pred(X, y, n_folds, model, *args, **kwargs):
    """
    Run n-fold cross validation returning a prediction for every row of X
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :param clf: The
    :param n_folds:
    :return:
    """
    # Construct a kfolds object
    kf = StratifiedKFold(y, n_folds=n_folds)
    y_pred = np.zeros(shape=y.shape)

    # Iterate through folds
    for train_index, test_index in kf:
        test = MLData(X[test_index], y[test_index])
        train = MLData(X[train_index], y[train_index])
        data = MLdataset(train, test)
        # Initialize a classifier with key word arguments
        model.fit(data)
        preds = model.predict(data)
        y_pred[test_index] = preds
    return y_pred


def remove_sparse_features(sparse_mat, threshold):
    """
    removes features (stars) with less than threshold observations in this data set
    :param X:
    :param threshold:
    :return: A version of X with columns that are too sparse removed and a list of the good column indices
    """
    print 'input matrix of shape: {0}'.format(sparse_mat.shape)
    observations = np.array(sparse_mat.sum(axis=0)).flatten()
    good_cols = np.where(observations >= threshold)[0]
    out_mat = sparse_mat[:, good_cols]
    print 'output matrix of shape: {0}'.format(out_mat.shape)
    return out_mat, good_cols


def edge_list_to_sparse_mat(edge_list):
    """
    Convert a pandas DF edge list into a scipy csc sparse matrix
    :param edge_list: A pandas DF with columns [fan_idx, star_idx]
    :return: A Columnar sparse matrix
    """
    # Create matrix representation (adjacency matrix) of edge list
    data_shape = edge_list.max(axis=0)
    print 'building sparse matrix of size {0}'.format(data_shape)
    X = lil_matrix((data_shape['fan_idx'] + 1, data_shape['star_idx'] + 1), dtype=int)
    X[edge_list['fan_idx'].values, edge_list['star_idx'].values] = 1
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


def remove_duplicate_labelled_fans():
    """
    Creates a deduplicated list of fans from the raw data
    :return:
    """
    fans = pd.read_csv('resources/raw_data/labelled_fans.csv')
    fans = fans.drop_duplicates('fan_id')
    fans[['fan_id', 'age']].to_csv('resources/labelled_fans.csv', index=False)


def preprocess_data(path):
    """
    Reads a csv with columns fan_id star_id star_idx num_followers cat weight
    Removes duplicates and creates and produces data in standard machine learning format X,y
    :param path: path to the training data
    :return: sparse csc matrix X of [fan_idx,star_idx]
    :return: numpy array y of target categories
    """
    temp = pd.read_csv(path)
    input_data = temp.drop_duplicates(['fan_id', 'star_id'])
    # replace the fan ids with an index
    fan_ids = input_data['fan_id'].drop_duplicates()
    idx = np.arange(len(fan_ids))
    lookup = pd.DataFrame(data={'fan_id': fan_ids.values, 'fan_idx': idx}, index=idx)
    all_data = input_data.merge(lookup, 'left')
    edge_list = all_data[['fan_idx', 'star_idx']]
    edge_list.columns = ['fan_idx', 'star_idx']
    y = all_data[['fan_idx', 'cat']].drop_duplicates()
    X = edge_list_to_sparse_mat(edge_list)
    return X, y, edge_list


def get_fan_idx_lookup():
    """
    Switch the fan_ids for indices - better for anonymity and making sparse matrics
    :return: writes resources/fan_list.csv
    """
    fans = pd.read_csv('resources/labelled_fans.csv')
    # The duplicates are quite error prone so it is possible to drop them all by setting
    # parameter keep=False this might cause problems with unindexed fans later though.
    fans = fans.drop_duplicates('fan_id')
    fans = fans.reset_index()
    fans[['index', 'fan_id']].to_csv('resources/fan_id2index_lookup.csv', index=False)


def pickle_sparse(sparse, path):
    """
    Writes a sparse matrix to disk in the python cPickle format
    :param sparse: A scipy s
    :param path:
    :return:
    """
    with open(path, 'wb') as outfile:
        pickle.dump(sparse, outfile, protocol=2)


def persist_edgelist(edge_list, path):
    """
    writes the edge_list to file as a .edgelist format file compatible with node2vec
    :param edge_list: A pandas DF with columns [fan_idx, star_idx]
    :param path: the path to write the file to
    :return: None
    """
    edge_list.to_csv(path, index=False, sep=" ", header=False)


def persist_data(folder, X, y):
    """
    Write the scipy csc sparse matrix X and a pandas DF y to disk
    :param path: the path to write data to
    :param X: scipy sparse css feature matrix
    :param y: pandas DF target values with columns [fan_idx, cat]
    :return: None
    """
    pickle_sparse(X, folder + '/X.p')
    y.to_pickle(folder + '/y.p')


def read_embedding(path, target, size):
    """
    Reads an embedding from text into a matrix
    :param path: the location of the embedding file
    :param size: the number of dimensions of the embedding eg. 64
    :param target: the target variables containing the indices to use
    :return:
    """
    data = pd.read_csv(path, header=None, index_col=0, skiprows=1, names=np.arange(size), sep=" ")
    # hack as I haven't made this bipartite yet
    data = data.ix[target['fan_idx']]
    # data = data.sort_index()
    # data = data.loc[0:6449, :]
    return data.as_matrix()


def read_pickle(path):
    with open(path, 'rb') as infile:
        return pickle.load(infile)


if __name__ == "__main__":
    X, y, edge_list = preprocess_data('resources/test.csv')
    persist_edgelist(edge_list, 'resources/test/test.edgelist')
    persist_data('resources/test', X, y)
