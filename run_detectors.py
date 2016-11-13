"""
Runs a set of candidate detectors on the age data
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, f_classif, RFECV
from sklearn import svm
import utils
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
import scipy.stats as stats

__author__ = 'benchamberlain'

names = [
    "Logistic_Regression",
    # "Nearest_Neighbors",
    # "Linear_SVM",
    # "RBF_SVM",
    # "Decision_Tree",
    # "Random_Forest"
    # "AdaBoost",
    # "Gradient_Boosted_Tree"
]

names64 = [
    "Logistic_Regression64",
    # "Nearest_Neighbors64",
    # "Linear_SVM64",
    # "RBF_SVM64",
    # "Decision_Tree64",
    # "Random_Forest64"
    # "AdaBoost64",
    # "Gradient_Boosted_Tree64"
]

names128 = [
    "Logistic_Regression128",
    # "Nearest_Neighbors128",
    # "Linear_SVM128",
    # "RBF_SVM128",
    # "Decision_Tree128",
    # "Random_Forest128"
    # "AdaBoost128",
    # "Gradient_Boosted_Tree128"
]

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=-1, max_iter=1000),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.0073),
    # SVC(kernel='rbf', gamma=0.011, C=9.0, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    # RandomForestClassifier(max_depth=18, n_estimators=50, criterion='gini', max_features=0.46, n_jobs=-1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_64 = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=-1, max_iter=1000),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.11),
    # SVC(kernel='rbf', gamma=0.018, C=31, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    # RandomForestClassifier(max_depth=6, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.21,n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_128 = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=-1, max_iter=1000),
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


def run_detectors(X, y, names, classifiers, n_folds):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """
    results = pd.DataFrame(np.zeros(shape=(len(names), n_folds)))
    results.index = names
    for name, detector in zip(names, classifiers):
        y_pred, results = run_cv_pred(X, y, detector, n_folds, name, results)
        print name
        utils.get_metrics(y, y_pred)
    return results


def run_cv_pred(X, y, clf, n_folds, name, results):
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
    for idx, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf.fit(X_train, y_train)
        try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
            preds = clf.predict(X_test)
        except TypeError:
            preds = clf.predict(X_test.todense())
        macro, micro = utils.get_metrics(preds, y[test_index])
        results.loc[name, idx] = macro
        y_pred[test_index] = preds

    return y_pred, results


def read_data(threshold):
    """
    reads the features and target variables
    :return:
    """
    x_path = 'resources/test/X_large.p'
    y_path = 'resources/test/y_large.p'
    X = utils.read_pickle(x_path)
    X1, cols = utils.remove_sparse_features(X, threshold=threshold)
    print X1.shape
    targets = utils.read_pickle(y_path)
    X2 = utils.read_embedding('local_resources/roberto_embeddings/user.factors.200.01reg.200iter', targets, size=200)
    y = np.array(targets['cat'])
    X3 = utils.read_embedding('local_resources/roberto_embeddings/user.factors.200.001reg.200iter', targets, size=200)
    # X3 = utils.read_embedding('resources/walks.emd', targets, size=64)
    X = [X1, X2, X3]
    return X, y


def stats_test(results):
    """
    performs a 2 sided t-test to see if difference in models is significant
    :param results:
    :return:
    """
    results['mean'] = results.mean(axis=1)
    results = results.sort('mean', ascending=False)

    print '1 versus 2'
    print(stats.ttest_ind(a=results.ix[0, 0:-1],
                          b=results.ix[1, 0:-1],
                          equal_var=False))
    print '2 versus 3'
    print(stats.ttest_ind(a=results.ix[1, 0:-1],
                          b=results.ix[2, 0:-1],
                          equal_var=False))

    print '3 versus 4'
    print(stats.ttest_ind(a=results.ix[1, 0:-1],
                          b=results.ix[2, 0:-1],
                          equal_var=False))

    return results


if __name__ == "__main__":
    X, y = read_data(5)
    print X[0].shape
    print y.shape
    n_folds = 5
    print 'without embedding'
    results = run_detectors(X[0], y, names, classifiers, n_folds)
    print results
    # print 'with 64 embedding'
    print 'their one'
    results64 = run_detectors(X[1], y, names64, classifiers_embedded_128, n_folds)
    # print 'with 128 embedding'
    print 'our one'
    results128 = run_detectors(X[2], y, names128, classifiers_embedded_128, n_folds)
    all_results = pd.concat([results, results64, results128])
    results = stats_test(all_results)
    print results
    outpath = 'results/roberto_emd/test' + utils.get_timestamp() + '.csv'
    results.to_csv(outpath, index=False)
    #
    # np.savetxt('y_pred.csv', y_pred, delimiter=' ', header='cat')
    # print accuracy(y, y_pred)
    #
    # unique, counts = np.unique(y_pred, return_counts=True)
    # print np.asarray((unique, counts)).T
