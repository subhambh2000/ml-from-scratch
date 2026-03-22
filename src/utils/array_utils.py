import numpy as np
import pandas as pd


def print_ndarray(arr, label=None):
    arr = np.asanyarray(arr)

    # python
    # print("DEBUG before if:", type(arr), "shape=", getattr(arr, "shape", None), "ndim=", getattr(arr, "ndim", None))
    # print("COND parts:", arr.ndim == 1, (arr.ndim == 2 and arr.shape[0] == 1), "combined:", bool(arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == 1)))

    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == 1):
        print(f"{label}: {np.array2string(arr.flatten(), separator=',')}")
    else:
        print(f"{label}:\n{pd.DataFrame(arr)}")