import numpy as np

def discretize_X_y(X: np.ndarray, y: np.ndarray, X_bins: int = 5, y_bins: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Discretizes the data and target. Produces equal distance intervals between the minimum and maximum value. Values are binned into these intervals.

    :param X: Data, containing only features.
    :type: np.ndarray
    :param y: Target vector.
    :type: np.ndarray
    :param X_bins: Bins produced for X. If 0, no discretization is performed. Default value is 5.
    :type: int
    :param y_bins: Bins produced for y. If 0, no discretization is performed. Default value is 2.
    :type: int
    :return: The discretized data and target variables.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    
    X_d = np.full(X.shape, np.nan)
    y_d = np.full(y.shape, np.nan)

    if X_bins != 0:
        X_sorted = np.sort(X, axis=0)
        position = np.full((X_bins, X.shape[1]), np.nan)
        for f in range(X.shape[1]):
            for i in range(1, X_bins + 1):
                position[i - 1, f] = X_sorted[int(X_sorted.shape[0] / X_bins * i) - 1, f]

        for i in range(X.shape[1]):
            X_d[:, i] = np.digitize(X[:, i], position[:, i], right=True).astype(int)

    else:
        X_d = X

    if y_bins != 0:
        y_sorted = np.sort(y)
        position = np.full(y_bins, np.nan)
        for i in range(1, y_bins + 1):
            position[i - 1] = y_sorted[int(y_sorted.shape[0] / y_bins * i) - 1]

        for i in range(y.shape[0]):
            y_d[i] = np.digitize(y[i], position, right=True).astype(int)

    else:
        y_d = y

    return X_d.astype(int), y_d.astype(int)