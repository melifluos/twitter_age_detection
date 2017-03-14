"""
Go from raw edge lists to sparse arrays of features with corresponding arrays of target variables
"""
import pandas as pd
import numpy as np
import utils


def preprocess_income_data():
    targets = pd.read_csv('local_resources/Socio_economic_classification_data/income_dataset/users-income', sep=' ')
    targets.columns = ['fan_id', 'mean_income']
    print 'target labels of shape: ', targets.shape
    edges = pd.read_csv('local_resources/Socio_economic_classification_data/income_dataset/users_friends.csv')
    edges.columns = ['fan_id', 'star_id']
    print 'edge list of shape: ', edges.shape
    all_data = edges.merge(targets)
    print 'all data of shape: ', all_data.shape
    X, y, edge_list = preprocess_data(all_data)
    utils.persist_edgelist(edge_list,
                           'local_resources/Socio_economic_classification_data/income_dataset/income.edgelist')
    utils.persist_data('local_resources/Socio_economic_classification_data/income_dataset/X.p',
                       'local_resources/Socio_economic_classification_data/income_dataset/y.p', X, y)


def preprocess_data(input_data):
    """
    Reads a csv with columns fan_id star_id star_idx num_followers cat weight
    Removes duplicates and creates and produces data in standard machine learning format X,y
    :param path: path to the training data
    :return: sparse csc matrix X of [fan_idx,star_idx]
    :return: numpy array y of target categories
    """
    input_data = input_data.drop_duplicates(['fan_id', 'star_id'])
    print 'input data of shape: ', input_data.shape, ' after duplicate removal'
    # remove known bad IDs
    input_data = utils.remove_bad_ids('resources/exclusion_list.csv', input_data)
    # replace the fan ids with an index
    fan_ids = input_data['fan_id'].drop_duplicates()
    fan_idx = np.arange(len(fan_ids))
    fan_lookup = pd.DataFrame(data={'fan_id': fan_ids.values, 'fan_idx': fan_idx}, index=fan_idx)
    with_fan_idx = input_data.merge(fan_lookup, 'left')
    print 'input data of shape: ', with_fan_idx.shape, ' after adding fan idx'
    # add star index
    star_ids = input_data['star_id'].drop_duplicates()
    star_idx = np.arange(len(star_ids))
    star_lookup = pd.DataFrame(data={'star_id': star_ids.values, 'star_idx': star_idx}, index=star_idx)
    all_data = with_fan_idx.merge(star_lookup, 'left')
    print 'input data of shape: ', all_data.shape, ' after adding star idx'
    edge_list = all_data[['fan_idx', 'star_idx']]
    edge_list.columns = ['fan_idx', 'star_idx']
    all_data.set_index('fan_id', inplace=True)
    y = all_data[['fan_idx', 'mean_income']].drop_duplicates()
    X = utils.edge_list_to_sparse_mat(edge_list)
    return X, y, edge_list


if __name__ == '__main__':
    preprocess_income_data()
