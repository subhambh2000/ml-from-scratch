from src.common.math_utils import mean, standard_deviation, _ensure_array


@_ensure_array
def standardize(arr):
    mean_value = mean(arr)
    std_dev_value = standard_deviation(arr)

    if std_dev_value == 0:
        return [0 for _ in arr]

    return [(x - mean_value) / std_dev_value for x in arr]
