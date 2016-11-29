"""
Perform parameter optimisation for the model
"""

from __future__ import division
from time import time
import numpy as np
from operator import itemgetter

from pybo import solve_bayesopt
from sklearn.neighbors import KNeighborsClassifier
from run_detectors import *
import utils
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform, expon
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# x_path = 'resources/X.p'
# y_path = 'resources/y.p'
# X = read_pickle(x_path)
# targets = read_pickle(y_path)
# y = np.array(targets['cat'])


def f1(x):
    clf = SVC(kernel="linear", C=x)
    pred = run_cv_pred(X, y, clf, n_folds=2)
    return f1_score(y, pred, average='macro')


def f2(x):
    clf = RandomForestClassifier(max_depth=5, n_estimators=20, criterion='entropy', max_features=x, n_jobs=-1)
    pred = run_cv_pred(X, y, clf, n_folds=2)
    return f1_score(y, pred, average='macro')


def f3(x):
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=x)
    pred = run_cv_pred(X, y, clf, n_folds=3)
    return f1_score(y, pred, average='macro')


def main():
    """Run the demo."""
    # grab a test function
    bounds = [0.001, 100]
    x = np.linspace(bounds[0], bounds[1], 1000)

    # solve the model
    xbest, model, info = solve_bayesopt(f3, bounds, niter=30, verbose=True)

    # make some predictions
    # mu, s2 = model.predict(x[:, None])
    #
    # # plot the final model
    # ax = figure().gca()
    # ax.plot_banded(x, mu, 2 * np.sqrt(s2))
    # ax.axvline(xbest)
    # ax.scatter(info.x.ravel(), info.y)
    # ax.figure.canvas.draw()
    # show()

    print xbest


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


svm_param_dist = {'C': expon(scale=100), 'gamma': expon(scale=.1),
                  'kernel': ['rbf'], 'class_weight': ['balanced', None]}


def run_pybo():
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    targets = utils.read_pickle(y_path)
    X1 = utils.read_pickle(x_path)
    X2 = utils.read_embedding('resources/test/balanced7_d64_window5.emd', targets, size=64)
    X3 = utils.read_embedding('resources/test/balanced7_window6.emd', targets, size=128)
    X = [X1, X2, X3]
    names = ['no embedding', '64 embedding', '128 embedding']
    y = np.array(targets['cat'])
    n_iter_search = 20
    clf = LogisticRegression(solver='lbfgs', n_jobs=1, max_iter=1000)
    logistic_param_dist = {'C': expon(scale=100), 'multi_class': ['ovr', 'multinomial']}
    scorer = make_scorer(f1_score, average='macro')
    random_search = RandomizedSearchCV(clf, param_distributions=logistic_param_dist,
                                       n_iter=n_iter_search, scoring=scorer)

    start = time()
    for i, features in enumerate(X):
        print names[i]
        random_search.fit(features, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.grid_scores_)


def run_RF_random_search():
    x_path = 'resources/test/X.p'
    y_path = 'resources/test/y.p'
    targets = utils.read_pickle(y_path)
    X1 = utils.read_pickle(x_path)
    X2 = utils.read_embedding('resources/test/test64.emd', targets, size=64)
    X3 = utils.read_embedding('resources/test/test128.emd', targets, size=128)
    X = [X1, X2, X3]
    names = ['no embedding', '64 embedding', '128 embedding']
    y = np.array(targets['cat'])
    # main(f1)
    n_iter_search = 20
    # specify parameters and distributions to sample from

    # randint takes a low and high val
    rf_param_dist = {"max_depth": sp_randint(2, 20),
                     "max_features": uniform(loc=0.01, scale=0.5),
                     # "min_samples_split": sp_randint(1, 11),
                     # "min_samples_leaf": sp_randint(1, 11),
                     "bootstrap": [True, False],
                     "criterion": ["gini", "entropy"]}
    clf = RandomForestClassifier(n_estimators=20, n_jobs=-1)
    # clf = SVC()
    scorer = make_scorer(f1_score, average='macro')
    random_search = RandomizedSearchCV(clf, param_distributions=rf_param_dist,
                                       n_iter=n_iter_search, scoring=scorer)

    start = time()
    for i, features in enumerate(X):
        print names[i]
        random_search.fit(features, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.grid_scores_)


def run_logistic_random_search():
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    targets = utils.read_pickle(y_path)
    X1 = utils.read_pickle(x_path)
    X2 = utils.read_embedding('resources/test/balanced7_d64_window5.emd', targets, size=64)
    X3 = utils.read_embedding('resources/test/balanced7_window6.emd', targets, size=128)
    X = [X1, X2, X3]
    names = ['no embedding', '64 embedding', '128 embedding']
    y = np.array(targets['cat'])
    n_iter_search = 20
    clf = LogisticRegression(solver='lbfgs', n_jobs=1, max_iter=1000)
    logistic_param_dist = {'C': expon(scale=100), 'multi_class': ['ovr', 'multinomial']}
    scorer = make_scorer(f1_score, average='macro')
    random_search = RandomizedSearchCV(clf, param_distributions=logistic_param_dist,
                                       n_iter=n_iter_search, scoring=scorer)

    start = time()
    for i, features in enumerate(X):
        print names[i]
        random_search.fit(features, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.grid_scores_)


if __name__ == '__main__':
    run_logistic_random_search()
