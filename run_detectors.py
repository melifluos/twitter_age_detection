"""
Runs a set of candidate detectors on the age data
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, f_classif, RFECV
from sklearn import svm
import utils
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
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
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=1.8),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.0073),
    # SVC(kernel='rbf', gamma=0.011, C=9.0, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=18, n_estimators=50, criterion='gini', max_features=0.46, n_jobs=-1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_64 = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=3.4),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.11),
    # SVC(kernel='rbf', gamma=0.018, C=31, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=6, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.21,n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_128 = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=3.9),
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
    skf = StratifiedKFold(n_splits=n_folds)
    splits = skf.split(X, y)
    y_pred = y.copy()

    # Iterate through folds
    for idx, (train_index, test_index) in enumerate(splits):
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


def run_all_datasets(datasets, y, names, classifiers, n_folds):
    """
    Loop through a list of datasets running potentially numerous classifiers on each
    :param datasets: iterable of numpy (sparse) arrays
    :param y: numpy (sparse) array of shape = (n_data, n_classes) of (n_data, 1)
    :param names: iterable of classifier names
    :param classifiers:
    :param n_folds:
    :return: A tuple of pandas DataFrames for each dataset containing (macroF1, microF1)
    """
    results = []
    for data in zip(datasets, names):
        temp = run_detectors(data[0], y, data[1], classifiers, n_folds)
        results.append(temp)
    return results


def read_roberto_embeddings(paths, target_path, sizes):
    targets = utils.read_pickle(target_path)
    y = np.array(targets['cat'])
    all_data = []
    for elem in zip(paths, sizes):
        data = utils.read_roberto_embedding(elem[0], targets, size=elem[1])
        all_data.append(data)
    return all_data, y


def read_embeddings(paths, target_path, sizes):
    targets = utils.read_pickle(target_path)
    y = np.array(targets['cat'])
    all_data = []
    for elem in zip(paths, sizes):
        data = utils.read_embedding(elem[0], targets, size=elem[1])
        all_data.append(data)
    return all_data, y


def build_ensembles(data, groups):
    """
    generates ensembles by columnwise concatenating arrays from a list
    :param data: A list of numpy arrays
    :param groups: a list of lists of indices into group. Each sub list represents the data sets to group together
    eg. [[1,2], [1,2,3]] will create 2 ensembles, the first containing the first and second data sets etc.
    :return: A list of numpy arrays where each array is an input to a classifier
    """
    ensemble_output = []
    for group in groups:
        ensemble = None
        for count, idx in enumerate(group):
            if count == 0:
                ensemble = data[idx - 1]
            else:
                ensemble = np.concatenate((ensemble, data[idx - 1]), axis=1)
        ensemble_output.append(ensemble)

    return ensemble_output


def roberto_scenario1():
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
    X, y = read_roberto_embeddings(paths, y_path, sizes)
    n_folds = 5
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/roberto_emd/age_large_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/roberto_emd/age_large_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def roberto_scenario2():
    paths = ['local_resources/roberto_embeddings/item.factors.200.0001reg.200iter',
             'local_resources/roberto_embeddings/item.factors.neg24',
             'local_resources/roberto_embeddings/item.factors.neg12']

    deepwalk_path = 'resources/test/test128_large.emd'

    names = [['logistic_0001reg.200iter'],
             ['logistic_neg24'],
             ['logistic_neg12'], ['logistic_deepwalk']]

    y_path = 'resources/test/y_large.p'

    target = utils.read_target(y_path)
    x_deepwalk = utils.read_embedding(deepwalk_path, target, 128)

    sizes = [201, 201, 201, 128]
    X, y = read_roberto_embeddings(paths, y_path, sizes)
    X.append(x_deepwalk)
    n_folds = 5
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/roberto_emd/age_large_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/roberto_emd/age_large_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def large_vs_small_scenario():
    deepwalk_path_large = 'resources/test/test128_large.emd'
    deepwalk_path = 'resources/test/test128.emd'

    names = [['logistic'], ['logistic_deepwalk']]
    names_large = [['logistic_large'], ['logistic_deepwalk_large']]

    y_path_large = 'resources/test/y_large.p'
    y_path = 'resources/test/y.p'

    x_path_large = 'resources/test/X_large.p'
    x_path = 'resources/test/X.p'

    target = utils.read_target(y_path)
    target_large = utils.read_target(y_path_large)

    x, y = utils.read_data(x_path, y_path, threshold=1)
    x_large, y_large = utils.read_data(x_path_large, y_path_large, threshold=1)

    x_deepwalk_large = utils.read_embedding(deepwalk_path_large, target_large, 128)
    x_deepwalk = utils.read_embedding(deepwalk_path, target, 128)

    n_folds = 5
    X = [x, x_deepwalk]
    X_large = [x_large, x_deepwalk_large]
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    results_large = run_all_datasets(X_large, y_large, names_large, classifiers, n_folds)
    all_results = utils.merge_results(results + results_large)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/age_small_v_large_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/age_small_v_large_micro' + utils.get_timestamp() + '.csv'
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
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/age_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/age_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def ensemble_scenario():
    deepwalk_path = 'resources/test/test128.emd'

    names = [['logistic'], ['logistic_deepwalk'], ['ensemble']]
    y_path = 'resources/test/y.p'
    x_path = 'resources/test/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=1)
    x_deepwalk = utils.read_embedding(deepwalk_path, target, 128)
    all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)

    n_folds = 5
    X = [x, x_deepwalk, all_features]
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/ensemble_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/ensemble_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced_ensemble_scenario(threshold):
    deepwalk_path = 'resources/test/balanced7.emd'

    names = [['logistic'], ['logistic_deepwalk'], ['ensemble']]
    y_path = 'resources/test/balanced7y.p'
    x_path = 'resources/test/balanced7X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=threshold)
    x_deepwalk = utils.read_embedding(deepwalk_path, target, 128)
    all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)

    n_folds = 5
    X = [x, x_deepwalk, all_features]
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_ensemble_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_ensemble_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced6_scenario():
    names = [['logistic']]
    y_path = 'resources/test/balanced6y.p'
    x_path = 'resources/test/balanced6X.p'

    target = utils.read_target(y_path)
    n_folds = 3
    x, y = utils.read_data(x_path, y_path, threshold=1)
    results = run_all_datasets([x], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced6_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced6_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_small_scenario():
    names = [['logistic']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'

    # target = utils.read_target(y_path)
    n_folds = 3
    x, y = utils.read_data(x_path, y_path, threshold=1)
    results = run_all_datasets([x], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_small_window_scenario():
    names = [['logistic']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    # y_path = 'resources/test/balanced7_100_thresh_y.p'
    y_path = 'resources/test/tempy.p'
    embedding_paths = []
    for i in np.arange(1, 10):
        embedding_paths.append('resources/test/balanced7_window' + str(i) + '.emd')
        names.append(['logistic_window' + str(i)])

    sizes = [128] * 10
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    ensembles = build_ensembles(x_emd, [[1, 2], [1, 2, 3]])
    n_folds = 3
    x, y = utils.read_data(x_path, y_path, threshold=1)
    X = x_emd[0:2] + [x_emd[5]] + ensembles
    new_names = names[0:2] + [names[6]] + [['ensemble_1_2'], ['ensemble_1_2_3']]
    results = run_all_datasets(X, y, new_names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_windows_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_windows_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_small_window_ensemble_scenario():
    names = []
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = []
    for i in np.arange(1, 10):
        embedding_paths.append('resources/test/balanced7_window' + str(i) + '.emd')
        names.append(['logistic_window' + str(i)])

    sizes = [128] * 10
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    ensembles = build_ensembles(x_emd, [[1, 2], [1, 2, 3]])
    n_folds = 5
    X = x_emd[0:3] + [x_emd[5]] + ensembles
    new_names = names[0:3] + [names[5]] + [['ensemble_1_2'], ['ensemble_1_2_3']]
    results = run_all_datasets(X, y, new_names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_windows_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_windows_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_small_window_64_128_scenario():
    names = []
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = []
    for i in np.arange(1, 10):
        embedding_paths.append('resources/test/balanced7_window' + str(i) + '.emd')
        names.append(['logistic_window128_' + str(i)])

    for i in np.arange(1, 10):
        embedding_paths.append('resources/test/balanced7_d64_window' + str(i) + '.emd')
        names.append(['logistic_window64_' + str(i)])

    sizes = [128] * 9 + [64] * 9
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    ensembles = build_ensembles(x_emd, [[10, 11], [10, 11, 12], [11, 13], [11, 15]])
    n_folds = 5
    X = x_emd + ensembles
    new_names = names + [['ensemble64_1_2'], ['ensemble64_1_2_3'], ['ensemble64_2_4'], ['ensemble64_2_6']]
    results = run_all_datasets(X, y, new_names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_windows_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_windows_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def two_step_scenario():
    det_names = [['1 step'], ['2 step']]
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = ['resources/test/balanced7_window10.emd', 'resources/test/balanced7_2step_window10.emd']
    sizes = [128] * 2
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    n_folds = 10
    results = run_all_datasets(x_emd, y, det_names, classifiers_embedded_128, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_window10_2step_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_window10_2step_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def LINE_scenario():
    names = [['logistic'], ['deepwalk'], ['LINE']]

    x_path = 'resources/test/balanced7_100_thresh_X.p'

    line_path = 'LINE/linux/vec_all.txt'
    y_path = 'resources/test/balanced7_100_thresh_y.p'

    targets = utils.read_pickle(y_path)

    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)

    deep_x = utils.read_embedding('resources/test/node2vec/1.0_1.0.emd', targets)

    line_x = utils.read_LINE_embedding(line_path, targets)
    results = run_all_datasets([x, deep_x, line_x], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_100_thresh_LINE_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_100_thresh_LINE_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_macro_LINE' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_micro_LINE' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_pq_scenario():
    names = [['logistic']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = []
    for p in [0.25, 0.5, 1.0, 2.0, 4.0]:
        for q in [0.25, 0.5, 1.0, 2.0, 4.0]:
            embedding_paths.append('resources/test/node2vec/' + str(p) + '_' + str(q) + '.emd')
            names.append(['logistic_window' + str(p) + '_' + str(q)])

    sizes = [128] * len(embedding_paths)
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)
    X = [x] + x_emd
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_pq_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_pq_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_pq_best_scenario():
    names = [['logistic'], ['logistic1.0_1.0'], ['logistic2.0_0.25']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = ['resources/test/node2vec/1.0_1.0.emd', 'resources/test/node2vec/2.0_0.25.emd']

    sizes = [128] * len(embedding_paths)
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)
    X = [x] + x_emd
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_100_thresh_pq_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_100_thresh_pq_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_pq_best_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_pq_best_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_normalisation_scenario():
    names = [['x'], ['axis 1 l2'], ['axis 0 l2'], ['axis 1 l1'], ['axis 0 l1'], ['emd'], ['emd axis 1 l2'],
             ['emd axis 0 l2'],
             ['emd axis 1 l1'], ['emd axis 0 l1']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    targets = utils.read_pickle(y_path)

    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)

    x1l2 = normalize(x, axis=1)
    x0l2 = normalize(x, axis=0)
    x1l1 = normalize(x, norm='l1', axis=1)
    x0l1 = normalize(x, norm='l1', axis=0)

    x_emd = utils.read_embedding('resources/test/node2vec/1.0_1.0.emd', targets)

    x_emd1l2 = normalize(x_emd, axis=1)
    x_emd0l2 = normalize(x_emd, axis=0)
    x_emd1l1 = normalize(x_emd, norm='l1', axis=1)
    x_emd0l1 = normalize(x_emd, norm='l1', axis=0)

    X = [x, x1l2, x0l2, x1l1, x0l1, x_emd, x_emd1l2, x_emd0l2, x_emd1l1, x_emd0l1]

    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_100_thresh_normalisation_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_100_thresh_normalisation_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_normalisation_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_normalisation_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_LINE_normalisation_scenario():
    names = [['1 deg no norm'], ['2 deg no norm'], ['1 deg feature l2'], ['2 deg feature l2'], ['ens'],
             ['ens l1 norm features'],
             ['ens l1 norm data'], ['ens l2 norm features'],
             ['ens l2 norm data'], ['deep line'], ['deepwalk']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    l1_path = 'LINE/linux/archive/vec_1st_wo_norm.txt'
    l2_path = 'LINE/linux/archive/vec_2nd_wo_norm.txt'
    targets = utils.read_pickle(y_path)

    x1 = utils.read_LINE_embedding(l1_path, targets)
    x2 = utils.read_LINE_embedding(l2_path, targets)

    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)

    x1a1l2 = normalize(x1, axis=1)
    x1a0l2 = normalize(x1, axis=0)
    x1a1l1 = normalize(x1, norm='l1', axis=1)
    x1a0l1 = normalize(x1, norm='l1', axis=0)

    x2a1l2 = normalize(x2, axis=1)
    x2a0l2 = normalize(x2, axis=0)
    x2a1l1 = normalize(x2, norm='l1', axis=1)
    x2a0l1 = normalize(x2, norm='l1', axis=0)

    # build ensembles
    ens = np.concatenate((x1, x2), axis=1)
    ens_a0l1 = np.concatenate((x1a0l1, x2a0l1), axis=1)
    ens_a1l1 = np.concatenate((x1a1l1, x2a1l1), axis=1)
    ens_a0l2 = np.concatenate((x1a0l2, x2a0l2), axis=1)
    ens_a1l2 = np.concatenate((x1a1l2, x2a1l2), axis=1)

    x_emd = utils.read_embedding('resources/test/node2vec/1.0_1.0.emd', targets)
    x_emd0l2 = normalize(x_emd, axis=0)
    deep_line = np.concatenate((x2a0l2, x_emd0l2), axis=1)

    X = [x1, x2, x1a0l2, x2a0l2, ens, ens_a0l1, ens_a1l1, ens_a0l2, ens_a1l2, deep_line, x_emd]

    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv(
        'results/age/balanced7_100_thresh_LINE_normalisation_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv(
        'results/age/balanced7_100_thresh_LINE_normalisation_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_LINE_normalisation_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_LINE_normalisation_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_normalised_ensemble_scenario():
    names = [['ensemb'], ['axis 0 l2'], ['emd axis 0 l2']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    targets = utils.read_pickle(y_path)
    l1_path = 'LINE/linux/archive/vec_1st_wo_norm.txt'
    l2_path = 'LINE/linux/archive/vec_2nd_wo_norm.txt'
    l1 = utils.read_LINE_embedding(l1_path, targets)
    l2 = utils.read_LINE_embedding(l2_path, targets)
    line1 = normalize(l1, axis=0)
    line2 = normalize(l1, axis=0)

    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)

    x1l2 = normalize(x, axis=1)
    x0l2 = normalize(x, axis=0)
    x1l1 = normalize(x, norm='l1', axis=1)
    x0l1 = normalize(x, norm='l1', axis=0)

    x_emd = utils.read_embedding('resources/test/node2vec/1.0_1.0.emd', targets)

    x_emd1l2 = normalize(x_emd, axis=1)
    x_emd0l2 = normalize(x_emd, axis=0)
    x_emd1l1 = normalize(x_emd, norm='l1', axis=1)
    x_emd0l1 = normalize(x_emd, norm='l1', axis=0)

    ensemb = np.concatenate((x0l2.toarray(), x_emd0l2), axis=1)
    ensemb_line = np.concatenate((line2, x_emd0l2), axis=1)
    X = [ensemb, ensemb_line, x0l2, x_emd0l2]

    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv(
        'results/age/balanced7_100_thresh_normalisation_deepwalk_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv(
        'results/age/balanced7_100_thresh_normalisation_deepwalk_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_normalisation_deepwalk_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_normalisation_deepwalk_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def tf_scenario():
    names = [['logistic'], ['deepwalk'], ['tf_deepwalk']]

    x_path = 'resources/test/balanced7_100_thresh_X.p'

    tf_path = 'resources/test/tf_test5.csv'
    y_path = 'resources/test/balanced7_100_thresh_y.p'

    targets = utils.read_pickle(y_path)

    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)

    deep_x = utils.read_embedding('resources/test/node2vec_1_1_test.emd', targets)

    tf_x = utils.read_tf_embedding(tf_path, targets)
    results = run_all_datasets([x, deep_x, tf_x], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_100_thresh_tf_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_100_thresh_tf_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_tf_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_tf_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_10_thresh_scenario():
    names = [['logistic'], ['l2 logistic'], ['deepwalk'], ['deepwalk_norm']]

    x_path = 'resources/test/balanced7_10_thresh_X.p'

    #tf_path = 'resources/test/tf_test5.csv'
    y_path = 'resources/test/balanced7_10_thresh_y.p'

    targets = utils.read_pickle(y_path)

    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=0)
    x_norm = normalize(x, axis=0)

    deep_x = utils.read_embedding('resources/test/balanced7_10_thresh.emd', targets)
    deep_x_norm = normalize(deep_x, axis=0)

    #tf_x = utils.read_tf_embedding(tf_path, targets)
    results = run_all_datasets([x, x_norm, deep_x, deep_x_norm], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_10_thresh_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_10_thresh_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_10_thresh_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_10_thresh_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def logistic_n_features_scenario():
    deepwalk_path = 'resources/test/balanced7.emd'

    y_path = 'resources/test/balanced7y.p'
    x_path = 'resources/test/balanced7X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=0)
    X = [normalize(x, axis=0)]
    names = [['0']]
    for thresh in xrange(5, 105, 5):
        features, _ = utils.remove_sparse_features(x, thresh)
        X.append(normalize(features, axis=0))
        names.append([str(thresh)])

    x_deepwalk = utils.read_embedding(deepwalk_path, target, 128)
    # all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)
    names.append(['deepwalk'])
    X.append(x_deepwalk)
    n_folds = 5
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_vary_nfeatures_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_vary_nfeatures_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_vary_nfeatures_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_vary_nfeatures__micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def karate_scenario():
    deepwalk_path = 'local_resources/zachary_karate/size8_walks1_len10.emd'

    y_path = 'local_resources/zachary_karate/y.p'
    x_path = 'local_resources/zachary_karate/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['logistic'], ['deepwalk']]

    x_deepwalk = utils.read_embedding(deepwalk_path, target)
    # all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)
    X = [x_deepwalk, normalize(x, axis=0)]
    n_folds = 2
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/karate/deepwalk_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/karate/deepwalk_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/karate/deepwalk_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/karate/deepwalk_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


if __name__ == "__main__":
    balanced7_10_thresh_scenario()
    # balanced7_pq_best_scenario()
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
