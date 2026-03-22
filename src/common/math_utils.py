import numpy as np


def is_list_or_array(arr):
    return isinstance(arr, (list, np.ndarray))


def is_matrix(arr):
    return (isinstance(arr, (list, np.ndarray)) and len(arr) > 0
            and all(isinstance(row, (list, np.ndarray)) for row in arr))


def _ensure_array(func):
    from functools import wraps
    @wraps(func)
    def wrapper(arr, *args, **kwargs):
        if not is_list_or_array(arr):
            return None

        if len(arr) == 1:
            return None

        return func(arr, *args, **kwargs)

    return wrapper


@_ensure_array
def mean(arr):
    size = len(arr)
    return sum(arr) / size


@_ensure_array
def variance(arr):
    mean_value = mean(arr)
    size = len(arr)
    squared_diffs = [(x - mean_value) ** 2 for x in arr]
    return sum(squared_diffs) / size


@_ensure_array
def standard_deviation(arr):
    return variance(arr) ** 0.5


def dot_product(v1, v2):
    if not is_list_or_array(v1) or not is_list_or_array(v2):
        return None

    if len(v1) != len(v2):
        return None

    elements_product = [a * b for a, b in zip(v1, v2)]
    return sum(elements_product)


def transpose(arr):
    if not is_matrix(arr):
        return None

    n = len(arr)
    m = len(arr[0])

    return [[arr[j][i] for j in range(n)] for i in range(m)]
