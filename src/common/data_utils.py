from math import floor
import numpy as np

from src.common.math_utils import is_matrix, is_list_or_array


def train_test_split(feature, target, test_size, shuffle=False):
    if not is_matrix(feature) or not is_list_or_array(target):
        return None

    n = len(feature)

    if n != len(target):
        return None

    if shuffle:
        indices = np.random.permutation(n)
        feature = np.asanyarray(feature)[indices]
        target = np.asanyarray(target)[indices]

    n_test = floor(n * test_size)
    n_train = n - n_test

    (x_train, x_test) = (feature[:n_train], feature[n_train:])
    (y_train, y_test) = (target[:n_train], target[n_train:])

    return x_train, y_train, x_test, y_test


