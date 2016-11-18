"""
Test a specific input data matrix. These test should not be run all of the time
"""
import numpy as np


def test_is_symmetric(X):
    rows, cols = X.shape
    assert rows == cols
    y = X.todense()
    y = np.array(y)
    assert np.array_equal(y, y.T)
