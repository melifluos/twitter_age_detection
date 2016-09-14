"""
Runs a set of candidate detectors on the age data
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, f_classif, RFECV
from sklearn import svm
from utils import *

__author__ = 'benchamberlain'


def run_detector(X, y, detector):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """


def run_cv_pred(X, y, clf, n_folds=10):
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
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


def accuracy(y, pred):
    return sum(y == pred) / len(y)


if __name__ == "__main__":
    x_path = 'local_resources/X.pkl'
    y_path = 'local_resources/y.pkl'
    X = read_pickle(x_path)
    y = read_pickle(y_path)
    clf = svm.LinearSVC()
    clf.fit(X)
    pred = clf.predict(y)

