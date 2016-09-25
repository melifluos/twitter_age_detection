"""
Perform parameter optimisation for the model
"""

from __future__ import division

import numpy as np

from ezplot import figure, show
from pybo import solve_bayesopt
from sklearn.neighbors import KNeighborsClassifier
from run_detectors import *

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
    bounds = [0.2, 1]
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


if __name__ == '__main__':
    main()
