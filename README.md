ML From Scratch
===============

Lightweight educational implementations of common machine learning building blocks in plain Python.
This repository contains small, easy-to-read implementations of numeric utilities, preprocessing helpers,
and basic metrics intended for learning and demonstration purposes.

Project status
--------------
This is an in-progress learning project. Several modules contain basic utilities (mean/variance/transpose/etc.)
and some modules are scaffolds for future algorithm implementations. A few functions intentionally return
`None` for invalid input; see function docs below.

Repository layout
-----------------
- `src/`
  - `common/`
    - `math_utils.py` — numeric helpers: `mean`, `variance`, `standard_deviation`, `dot_product`, `transpose`, helper decorators and validators.
    - `preprocessing.py` — preprocessing functions: `standardize` (uses `math_utils._ensure_array`).
    - `metrics.py` — evaluation metrics: `mean_squared_error`.
    - `data_utils.py` — dataset helpers: `train_test_split`.
  - `utils/`
    - `array_utils.py` — small helper to print numpy arrays in a readable way (`print_ndarray`).
  - algorithm packages (placeholders / scaffolding):
    - `decision_tree/`
    - `knn/`
    - `linear_regression/`
    - `logistic_regression/`
    - `neural_network/`
- `tests/` — test folder (may contain unit tests; run with pytest if present).
- `requirements.txt` — Python dependencies (install with pip).

Key modules and API
-------------------

`src/common/math_utils.py`
- `is_list_or_array(arr)` -> bool  
  Returns True if `arr` is a `list` or `numpy.ndarray`.
- `is_matrix(arr)` -> bool  
  True when `arr` is a 2D structure (list/ndarray of rows).
- `_ensure_array` (decorator)  
  Internal decorator used by statistical functions. It enforces that the argument is a list/ndarray and has length > 1; otherwise the wrapped function returns `None`.
- `mean(arr)` -> float | None  
  Arithmetic mean. Returns `None` if `arr` is not a list/array or has length <= 1 (because of `_ensure_array`).
- `variance(arr)` -> float | None  
  Population variance (divides by N). Uses `mean`.
- `standard_deviation(arr)` -> float | None  
  Square root of `variance`.
- `dot_product(v1, v2)` -> float | None  
  Returns dot product for two vectors of equal length. Returns `None` for invalid inputs or mismatched lengths.
- `transpose(arr)` -> list[list] | None  
  Returns transposed matrix as nested lists, or `None` if input is not matrix-like.

Notes: Many functions validate inputs and return `None` rather than raising exceptions for invalid inputs.

`src/common/preprocessing.py`
- `standardize(arr)` -> list | None  
  Standardizes a 1D array: (x - mean) / std. If the standard deviation is zero, returns a list of zeros.
  The decorator `_ensure_array` is used here as well, so `None` will be returned for invalid inputs or arrays of length 1.

`src/common/metrics.py`
- `mean_squared_error(v1, v2)` -> float | None  
  Computes MSE between two sequences of equal length. Returns `None` for invalid inputs.

`src/utils/array_utils.py`
- `print_ndarray(arr, label=None)`  
  Nicely prints numpy arrays: prints single-row or 1D arrays compactly; uses `pandas.DataFrame` for other arrays.

`src/common/data_utils.py`
- `train_test_split(feature, target, test_size, shuffle=False)` Implemented. 
- Behavior:
  - Validates inputs: features must be a two-dimensional sequence and targets a one-dimensional sequence; returns None on invalid input.
  - Requires equal length between feature rows and target entries.
  - Optional shuffling preserves row-target pairing and converts inputs to array-like types when applied.
  - Test set size is computed as the floor of n multiplied by test_size; training size is the remainder.
  - Returns four items in this order: training features, training targets, test features, test targets — this order differs from some common libraries.


Development notes & TODOs
------------------------
- `train_test_split` is implemented in `src/common/data_utils.py`. Consider the following improvements:
    - Make the signature more flexible and sklearn-compatible: `train_test_split(features, targets, test_size=0.2, shuffle=True, random_state=None)`.
    - Consider reordering the return to `(X_train, X_test, y_train, y_test)` to match common expectations or clearly document the current order.
    - Add an optional `random_state` (seed) for reproducible shuffling.
    - Add input type hints, docstrings and explicit exceptions (instead of returning `None`) for clearer failure modes.
    - Add unit tests that assert shapes/lengths, behavior on edge-cases (small n, test_size edge values, zero variance), and shuffling reproducibility.
- Add algorithm implementations into the scaffolded packages: `knn`, `linear_regression`, `logistic_regression`, `decision_tree`, `neural_network`.
- Consider replacing or augmenting the current input validation with explicit exceptions for clearer failures.
- Add type hints and docstrings across modules for better developer experience.
- Consider adding CI (GitHub Actions) to run linting and tests automatically.
- Add packaging files (`pyproject.toml` or `setup.py`) for easier installs and imports.

Design & behavior notes (important)
----------------------------------
- Several functions use defensive input validation and return `None` when inputs are invalid. 
- This is a design choice in the current codebase — be mindful to check for `None` when calling these functions.
- `variance` and `standard_deviation` compute population statistics (division by N, not N-1).
- `standardize` returns a list (not numpy array).

Contribution
------------
Contributions welcome. Recommended workflow:
1. Fork the repository.
2. Create a feature branch.
3. Make changes and run tests.
4. Open a pull request with a clear description.

License
-------
Add a license of your choice (MIT, Apache 2.0, BSD, etc.). This README currently does not include a license — add one in the repo root (`LICENSE`) to make the project reusable.


Acknowledgements
----------------
This project is an educational "from scratch" implementation of ML basics and utilities for demonstration and learning.