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
    # RandomForestClassifier(max_depth=6, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.21,n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_128 = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=-1, max_iter=1000),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.11),
    # SVC(kernel='rbf', gamma=0.029, C=27.4, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    # RandomForestClassifier(max_depth=7, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.12,n_jobs = -1),
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
    temp = pd.DataFrame(np.zeros(shape=(len(names), n_folds)))
    temp.index = names
    results = (temp, temp.copy())
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
        results[0].loc[name, idx] = macro
        results[1].loc[name, idx] = micro
        y_pred[test_index] = preds

    return y_pred, results


def read_data(threshold, size):
    """
    reads the features and target variables
    :return:
    """
    x_path = 'resources/test/X_large.p'
    y_path = 'resources/test/y_large.p'
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    X = utils.read_pickle(x_path)
    X1, cols = utils.remove_sparse_features(X, threshold=threshold)
    print X1.shape
    X2 = utils.read_embedding('local_resources/roberto_embeddings/item.factors.200.01reg.200iter', targets, size=size)
    X3 = utils.read_embedding('local_resources/roberto_embeddings/item.factors.200.001reg.200iter', targets, size=size)
    # X3 = utils.read_embedding('resources/walks.emd', targets, size=64)
    X = [X1, X2, X3]
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
    for data in zip(datasets, names):
        temp = run_detectors(data[0], y, data[1], classifiers, n_folds)
        results.append(temp)
    return results


def read_embeddings(paths, target_path, sizes):
    targets = utils.read_pickle(target_path)
    y = np.array(targets['cat'])
    all_data = []
    for elem in zip(paths, sizes):
        data = utils.read_roberto_embedding(elem[0], targets, size=elem[1])
        all_data.append(data)
    return all_data, y


def roberto_scenario():
    paths = ['local_resources/roberto_embeddings/item.factors.200.01reg.200iter',
             'local_resources/roberto_embeddings/item.factors.200.001reg.200iter',
             'local_resources/roberto_embeddings/item.factors.200.0001reg.200iter',
             'local_resources/roberto_embeddings/item.factors.200.00001reg.200iter',
             'local_resources/roberto_embeddings/item.factors.200.noreg.200iter']

    names = [['logistic_01reg.200iter'],
             ['logistic_001reg.200iter'],
             ['logistic_0001reg.200iter'],
             ['logistic_00001reg.200iter'],
             ['logistic_noreg.200iter']]

    y_path = 'resources/test/y_large.p'

    sizes = [201, 201, 201, 201, 200]
    X, y = read_embeddings(paths, y_path, sizes)
    n_folds = 5
    results = utils.run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/roberto_emd/age_large_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/roberto_emd/age_large_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def bipartite_scenario():
    paths = ['resources/test/test128.emd', 'resources/test/test1282.emd']

    names = [['logistic_theirs'],
             ['logistic_mine']]

    y_path = 'resources/test/y.p'

    sizes = [128, 128]
    X, y = read_embeddings(paths, y_path, sizes)
    n_folds = 5
    results = utils.run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/age_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/age_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def blogcatalog_scenario():
    paths = ['local_resources/blogcatalog/X.p']

    names = [['logistic']]

    y_path = 'local_resources/blogcatalog/y.p'

    sizes = [128]
    X, y = read_data(paths, y_path, sizes)
    n_folds = 5
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/blogcatalog/debug_test_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/blogcatalog/debug_test_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


if __name__ == "__main__":
    bipartite_scenario()

# size = 201
# X, y = read_data(5, size)
# print X[0].shape
# print y.shape
# n_folds = 5
# print 'without embedding'
# results = run_detectors(X[0], y, names, classifiers, n_folds)
# print results
# # print 'with 64 embedding'
# print 'their one'
# results64 = run_detectors(X[1], y, names64, classifiers_embedded_128, n_folds)
# # print 'with 128 embedding'
# print 'our one'
# results128 = run_detectors(X[2], y, names128, classifiers_embedded_128, n_folds)
# all_results = merge_results([results, results64, results128])

# np.savetxt('y_pred.csv', y_pred, delimiter=' ', header='cat')
# print accuracy(y, y_pred)
#
# unique, counts = np.unique(y_pred, return_counts=True)
# print np.asarray((unique, counts)).T
