"""
Runs a set of candidate detectors on the age data
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, f_classif, RFECV
from sklearn import svm
from utils import *
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

__author__ = 'benchamberlain'

names = [
    "Logistic Regression",
    # "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    # "Decision Tree",
    "Random Forest"
    # "AdaBoost",
    # "Gradient Boosted Tree"
]

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='sag', n_jobs=-1, max_iter=1000),
    # KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.0073),
    SVC(kernel='rbf', gamma=0.011, C=9.0, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    RandomForestClassifier(max_depth=18, n_estimators=50, criterion='gini', max_features=0.46, n_jobs=-1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_64 = [
    LogisticRegression(multi_class='multinomial', solver='sag', n_jobs=-1, max_iter=1000),
    # KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.11),
    SVC(kernel='rbf', gamma=0.018, C=31, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    RandomForestClassifier(max_depth=6, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.21,
                           n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_128 = [
    LogisticRegression(multi_class='multinomial', solver='sag', n_jobs=-1, max_iter=1000),
    # KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.11),
    SVC(kernel='rbf', gamma=0.029, C=27.4, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    RandomForestClassifier(max_depth=7, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.12,
                           n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

def run_detectors(X, y, classifiers):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """
    for name, detector in zip(names, classifiers):
        y_pred = run_cv_pred(X, y, detector)
        print name
        get_metrics(y, y_pred)


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

if __name__ == "__main__":
    x_path = 'resources/test/X.p'
    y_path = 'resources/test/y.p'
    X = read_pickle(x_path)
    X1, cols = remove_sparse_features(X, threshold=2)
    print X1.shape
    targets = read_pickle(y_path)
    X2 = read_embedding('resources/test/test64.emd', targets, size=64)
    y = np.array(targets['cat'])
    print 'without embedding'
    run_detectors(X1, y, classifiers)
    print 'with 64 embedding'
    run_detectors(X2, y, classifiers_embedded_64)
    print 'with 128 embedding'
    X3 = read_embedding('resources/test/test128.emd', targets, size=128)
    run_detectors(X3, y, classifiers_embedded_128)
    #
    # np.savetxt('y_pred.csv', y_pred, delimiter=' ', header='cat')
    # print accuracy(y, y_pred)
    #
    # unique, counts = np.unique(y_pred, return_counts=True)
    # print np.asarray((unique, counts)).T
