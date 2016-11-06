"""
Evaluation of the keras high level deep learning library
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder

__author__ = 'benchamberlain'


class keras_model:
    """
    wrapper for a keras model so it fits in a scikit-learn shaped hole
    """

    def __init__(self, model, batch_size, training_epochs):
        self.batch_size = batch_size
        self.epoch = training_epochs
        self.model = model

    def fit(self, data):
        """
        fit the model
        :param data is an MLDataset object
        """
        x = data.train.features
        y = data.train.target
        target = one_hot(y)
        self.model.fit(x, target, batch_size=self.batch_size, nb_epoch=self.epoch)

    def predict(self, data):
        """
        predict unseen test values
        :param data is an MLDataset object
        """
        x = data.test.features
        return self.model.predict_classes(x, self.batch_size)


def read_data(threshold):
    """
    Read the Twitter user ages test data. The target variables must be a one-hot-encoded
    matrix with dimensions [n_data, 10]
    :param threshold: the minimum number of edges each account must have
    :return: sparse feature matrix and dense np.array target matrix
    """
    x_path = 'resources/test/X.p'
    y_path = 'resources/test/y.p'
    X = utils.read_pickle(x_path)
    features, cols = utils.remove_sparse_features(X, threshold=threshold)
    # features = np.array(X.todense())
    targets = utils.read_pickle(y_path)
    target = np.array(targets['cat'])
    unique, counts = np.unique(target, return_counts=True)
    total = float(sum(counts))
    norm_counts = counts / total
    print(np.asarray((unique, norm_counts)).T)
    return np.array(features.todense()), target


def one_hot(y):
    """
    one hot encode the target variables
    :param y: a 1d numpy array
    :return: a [n_data, 10] numpy array
    """
    enc = OneHotEncoder(n_values=10, sparse=False)
    temp = enc.fit_transform(y)
    return np.reshape(temp, newshape=[-1, 10], order='C')


def build_model(input_dim):
    """
    build a keras DL model
    :return: a Sequential object
    """
    model = Sequential()

    model.add(Dense(output_dim=128, input_dim=input_dim))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    X, y = read_data(threshold=5)
    print X.shape
    print y.shape
    n_data, n_features = X.shape
    model = build_model(n_features)
    km = keras_model(model, batch_size=32, training_epochs=1)
    y_preds = utils.run_cv_pred(X, y, n_folds=2, model=km)
    print type(y_preds)
    print y_preds.shape
    print y_preds[0:10]
    print y[0:10]
    print utils.get_metrics(y, y_preds)
    # print loss_and_metrics
