"""
Run multilabel classification on networked datasets
"""

__author__ = 'benchamberlain'

"""
Runs a set of candidate detectors on the age data
"""
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import utils
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import multilabel_evaluation as mle
import scipy.stats as stats
import datetime

__author__ = 'benchamberlain'

names = [
    "Logistic_Regression",
    # "Nearest_Neighbors",
    "Linear_SVM",
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
    "Linear_SVM128",
    # "RBF_SVM128",
    # "Decision_Tree128",
    # "Random_Forest128"
    # "AdaBoost128",
    # "Gradient_Boosted_Tree128"
]

classifiers = [
    OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=1, max_iter=1000), n_jobs=1),
    # KNeighborsClassifier(3),
    OneVsRestClassifier(SVC(kernel="linear", C=0.0073, probability=True)),
    # SVC(kernel='rbf', gamma=0.011, C=9.0, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    # RandomForestClassifier(max_depth=18, n_estimators=50, criterion='gini', max_features=0.46, n_jobs=-1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_64 = [
    OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=1, max_iter=1000), n_jobs=1),
    # KNeighborsClassifier(3),
    # OneVsRestClassifier(SVC(kernel="linear", C=0.11, probability=True)),
    # SVC(kernel='rbf', gamma=0.018, C=31, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    # RandomForestClassifier(max_depth=6, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.21,n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_128 = [
    OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=1, max_iter=1000), n_jobs=1),
    # KNeighborsClassifier(3),
    OneVsRestClassifier(SVC(kernel="linear", C=0.11, probability=True)),
    # SVC(kernel='rbf', gamma=0.029, C=27.4, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    # RandomForestClassifier(max_depth=7, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.12,n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]


def run_detectors(X, y, names, classifiers, n_folds):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix of shape=(n_data, n_features)
    :param y: A scipy sparse multilabel array of shape=(n_data, n_labels)
    :param names: A list of detector names being run
    :param classifiers: a list of classifiers
    :param n_folds: the number of splits of the data to make
    :return: The accuracy of the detector
    """
    temp = pd.DataFrame(np.zeros(shape=(len(names), n_folds)))
    temp.index = names
    results = (temp, temp.copy())
    for name, detector in zip(names, classifiers):
        print name
        results = run_cv_pred(X, y, detector, n_folds, name, results)
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
    kf = KFold(y.shape[0], n_folds=n_folds)
    y_pred = y.copy()

    # Iterate through folds
    for idx, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf.fit(X_train, y_train)
        try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
            probs = clf.predict_proba(X_test)
        except TypeError:
            probs = clf.predict_proba(X_test.todense())
        macro, micro = mle.evaluate(probs, y[test_index])
        print 'macro F1, micro F1', macro, micro
        results[0].loc[name, idx] = macro
        results[1].loc[name, idx] = micro
        # y_pred[test_index] = preds

    return results


def read_data(target_path, feature_path, embedding_paths):
    """
    Read in a public data set and associated embeddings
    :param target_path: target values / labels
    :param feature_path: path to a pickled sparse matrix
    :param embedding_paths: path to the embeddings files
    :return: A list of feature arrays, an array of target values
    """
    X = [utils.read_pickle(feature_path)]
    for path in embedding_paths:
        X.append(utils.read_public_embedding(path, 128))
    y = utils.read_pickle(target_path)
    # print y.shape
    # print 'n double labels'
    # sums = y.sum(axis=1)
    # print y[:].sum()
    # df = pd.DataFrame(data=y.todense())
    # df['sums'] = sums
    # df.to_csv('local_resources/blogcatalog/ytest.csv', index=False, header=None)
    return X, y


def run_all_datasets(datasets, y, names, classifiers, n_folds):
    """
    Loop through a list of datasets running potentially numerous classifiers on each
    :param datasets:
    :param y:
    :param names:
    :param classifiers:
    :param n_folds:
    :return: A tuple of pandas DataFrames for each dataset containing (macroF1, microF1)
    """
    results = []
    for data in zip(datasets, names, classifiers):
        temp = run_detectors(data[0], y, data[1], data[2], n_folds)
        results.append(temp)
    return results


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


def blogcatalog_scenario():
    target_path = 'local_resources/blogcatalog/y.p'
    feature_path = 'local_resources/blogcatalog/X.p'
    embedding_paths = ['local_resources/blogcatalog/blogcatalog128.emd']
    classifier_names = [names, names128]
    detectors = [classifiers,
                 classifiers_embedded_128]
    X, y = read_data(target_path, feature_path, embedding_paths)
    n_folds = 5
    results = run_all_datasets(X, y, classifier_names, detectors, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/blogcatalog/macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/blogcatalog/micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)

    # print 'without embedding'
    # results = run_detectors(X[0], y, names, classifiers, n_folds)
    # # print results
    # # print 'with 64 embedding'
    # print 'with embedding'
    # # y = y[0:100, :]
    # # X1 = X[1][0:100, :]
    # results64 = run_detectors(X[1], y, names64, classifiers_embedded_128, n_folds)
    # # print 'with 128 embedding'
    # # print 'our one'
    # # results128 = run_detectors(X[2], y, names128, classifiers_embedded_128, n_folds)
    # all_results = pd.concat([results, results64])
    # results = stats_test(all_results)
    # print results
    # outpath = 'results/blogcatalog/debug_test' + utils.get_timestamp() + '.csv'
    # results.to_csv(outpath, index=True)


if __name__ == "__main__":
    s = datetime.datetime.now()
    blogcatalog_scenario()
    print datetime.datetime.now() - s
    # X, y = read_data(5)

    #
    # np.savetxt('y_pred.csv', y_pred, delimiter=' ', header='cat')
    # print accuracy(y, y_pred)
    #
    # unique, counts = np.unique(y_pred, return_counts=True)
    # print np.asarray((unique, counts)).T
