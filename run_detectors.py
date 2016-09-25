"""
Runs a set of candidate detectors on the age data
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, f_classif, RFECV
from sklearn import svm
from utils import *
from sklearn.cross_validation import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

__author__ = 'benchamberlain'

names = [
    # "Nearest Neighbors",
    "Linear SVM",
    # "RBF SVM",
    # "Decision Tree",
    "Random Forest"
    # "AdaBoost",
    # "Gradient Boosted Tree"
]

classifiers = [
    # KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.0073),
    # SVC(gamma=2, C=1),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    RandomForestClassifier(max_depth=5, n_estimators=20, criterion='entropy', max_features=0.1, n_jobs=-1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]


def run_detectors(X, y):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """
    for name, detector in zip(names, classifiers):
        y_pred = run_cv_pred(X, y, detector)
        print name
        print accuracy(y, y_pred)


def run_cv_pred(X, y, clf, n_folds=3):
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
        try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
            y_pred[test_index] = clf.predict(X_test)
        except TypeError:
            y_pred[test_index] = clf.predict(X_test.todense())

    return y_pred


def accuracy(y, pred):
    return sum(y == pred) / float(len(y))


if __name__ == "__main__":
    x_path = 'resources/X.p'
    y_path = 'resources/y.p'
    X = read_pickle(x_path)
    targets = read_pickle(y_path)
    y = np.array(targets['cat'])
    y_pred = run_detectors(X, y)
    #
    # np.savetxt('y_pred.csv', y_pred, delimiter=' ', header='cat')
    # print accuracy(y, y_pred)
    #
    # unique, counts = np.unique(y_pred, return_counts=True)
    # print np.asarray((unique, counts)).T
