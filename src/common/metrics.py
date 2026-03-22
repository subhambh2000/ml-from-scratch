from src.common.math_utils import is_list_or_array


def mean_squared_error(v1, v2):
    if not is_list_or_array(v1) or not is_list_or_array(v2):
        return None

    size = len(v1)
    if size != len(v2):
        return None

    squared_diff = [(x - y) ** 2 for x, y in zip(v1, v2)]
    return sum(squared_diff) / size
