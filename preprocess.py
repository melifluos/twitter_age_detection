"""
Go from raw edge lists to sparse arrays of features with corresponding arrays of target variables
"""
import pandas as pd
import numpy as np
import utils


def preprocess_income_data():
    targets = pd.read_csv('local_resources/Socio_economic_classification_data/income_dataset/users-income')
    targets.columns = ['fan_id', 'mean_income']
    print 'target labels of shape: ', targets.shape
    edges = pd.read_csv('local_resources/Socio_economic_classification_data/income_dataset/users_friends.csv')
    edges.columns = ['fan_id', 'star_id']
    print 'edge list of shape: ', edges.shape
    X, y, edge_list = preprocess_data(edges, targets)
    utils.persist_edgelist(edge_list, 'resources/test/balanced7.edgelist')
    utils.persist_data('resources/test/balanced7X.p', 'resources/test/balanced7y.p', X, y)


def preprocess_data(edges, targets):
    """
    Reads a csv with columns fan_id star_id star_idx num_followers cat weight
    Removes duplicates and creates and produces data in standard machine learning format X,y
    :param path: path to the training data
    :return: sparse csc matrix X of [fan_idx,star_idx]
    :return: numpy array y of target categories
    """
    input_data = edges.drop_duplicates(['fan_id', 'star_id'])
    # remove known bad IDs
    input_data = utils.remove_bad_ids('resources/exclusion_list.csv', input_data)
    # replace the fan ids with an index
    fan_ids = input_data['fan_id'].drop_duplicates()
    idx = np.arange(len(fan_ids))
    fan_lookup = pd.DataFrame(data={'fan_id': fan_ids.values, 'fan_idx': idx}, index=idx)
    all_data = input_data.merge(fan_lookup, 'left')
    fan_ids = input_data['fan_id'].drop_duplicates()
    idx = np.arange(len(fan_ids))
    fan_lookup = pd.DataFrame(data={'fan_id': fan_ids.values, 'fan_idx': idx}, index=idx)
    all_data = input_data.merge(fan_lookup, 'left')
    edge_list = all_data[['fan_idx', 'star_idx']]
    edge_list.columns = ['fan_idx', 'star_idx']
    y = all_data[['fan_idx', 'cat']].drop_duplicates()
    X = edge_list_to_sparse_mat(edge_list)
    return X, y, edge_list
