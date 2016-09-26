"""
Perform parameter optimisation for the model
"""

from __future__ import division
from time import time
import numpy as np
from operator import itemgetter

from ezplot import figure, show
from pybo import solve_bayesopt
from sklearn.neighbors import KNeighborsClassifier
from run_detectors import *
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform


x_path = 'resources/X.p'
y_path = 'resources/y.p'
X = read_pickle(x_path)
targets = read_pickle(y_path)
y = np.array(targets['cat'])


def f1(x):
    """
    Test function that we will optimize. This is a simple sinusoidal function
    whose maximum should be found very quickly.
    """
    clf = SVC(kernel="linear", C=x)
    pred = run_cv_pred(X, y, clf, n_folds=2)
    return accuracy(y, pred)


def f2(x):
    """
    Test function that we will optimize. This is a simple sinusoidal function
    whose maximum should be found very quickly.
    """
    clf = RandomForestClassifier(max_depth=5, n_estimators=20, criterion='entropy', max_features=x, n_jobs=-1)
    pred = run_cv_pred(X, y, clf, n_folds=2)
    return accuracy(y, pred)


def main():
    """Run the demo."""
    # grab a test function
    bounds = [0.001, 0.15]
    x = np.linspace(bounds[0], bounds[1], 500)

    # solve the model
    xbest, model, info = solve_bayesopt(f2, bounds, niter=30, verbose=True)

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


# specify parameters and distributions to sample from
# randint takes a low and high val
param_dist = {"max_depth": sp_randint(2, 11),
              "max_features": uniform(loc=0.01, scale=0.05),
              #"min_samples_split": sp_randint(1, 11),
              #"min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}



if __name__ == '__main__':
    # run randomized search
    n_iter_search = 20
    clf = RandomForestClassifier(n_estimators=20, n_jobs=-1)
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)
    # main()
