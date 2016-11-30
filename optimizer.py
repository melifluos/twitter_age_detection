"""
Perform parameter optimisation for the model
"""

from __future__ import division
from time import time
import numpy as np
from operator import itemgetter

from pybo import solve_bayesopt
from sklearn.neighbors import KNeighborsClassifier
import utils
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform, expon
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

x_path = 'resources/test/balanced7_100_thresh_X.p'
y_path = 'resources/test/balanced7_100_thresh_y.p'
targets = utils.read_pickle(y_path)
X = utils.read_pickle(x_path)
# X2 = utils.read_embedding('resources/test/balanced7_d64_window5.emd', targets, size=64)
# X3 = utils.read_embedding('resources/test/balanced7_window6.emd', targets, size=128)
# X = [X1, X2, X3]
# names = ['no embedding', '64 embedding', '128 embedding']
y = np.array(targets['cat'])


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


def f4(x):
    clf = LogisticRegression(multi_class='multinomial', C=x[0])
    pred = run_cv_pred(X, y, clf, n_folds=2)
    return f1_score(y, pred, average='macro')


def run_cv_pred(X, y, clf, n_folds):
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
        y_pred[test_index] = preds

    return y_pred


def configure_pybo(X, y):
    """
    A function that returns a classifier function with a single argument to be optimized
    :param X: features
    :param y: targets
    :return: function to be optimised
    """

    def f(x):
        # the pybo code passes x as an array, which scikit-learn doesn't like, hence x[0]
        detector = LogisticRegression(solver='lbfgs', n_jobs=1, max_iter=1000, C=x[0])
        pred = run_cv_pred(X, y, detector, n_folds=3)
        return f1_score(y, pred, average='macro')

    return f


# def plot_pybo(info);
#     make some predictions
#     mu, s2 = model.predict(x[:, None])
#
#     # plot the final model
#     ax = figure().gca()
#     ax.plot_banded(x, mu, 2 * np.sqrt(s2))
#     ax.axvline(xbest)
#     ax.scatter(info.x.ravel(), info.y)
#     ax.figure.canvas.draw()
#     show()


def main():
    """Run the demo."""
    # grab a test function
    bounds = [0.1, 100]
    x = np.linspace(bounds[0], bounds[1], 1000)

    # solve the model
    xbest, model, info = solve_bayesopt(f1, bounds, niter=30, verbose=True)

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

    logistic_param_dist = {'C': expon(scale=100), 'multi_class': ['ovr', 'multinomial']}

    start = time()
    for i, features in enumerate(X):
        print names[i]
        f = configure_pybo(features, y)
        bounds = np.array([0.0, 100.0])
        xbest, model, info = solve_bayesopt(f, bounds.T, niter=30, verbose=True)
        print("Bayesian optimisation took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        print xbest


def run_RF_random_search():
    """
    Perform scikit-learn random search over random forest parameters
    :return:
    """
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
    # run_pybo()
    main()
