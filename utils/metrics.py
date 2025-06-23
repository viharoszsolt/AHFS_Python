from collections.abc import Callable
import numpy as np

from sklearn.metrics import accuracy_score

class Metrics:
    @property
    def _metric_modes(self) -> dict[Callable[[], float], str]:
        return {
            self.accuracy_oh: "one-hot",
            self.accuracy_reg: "normal",
            self.root_mean_squared_oh: "one-hot",
            self.scaled_root_mean_squared_oh: "one-hot",
            self.root_mean_squared: "normal",
            self.scaled_root_mean_squared: "normal"
        }

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, classes: None|np.ndarray = None):
        """
        A collection of metrics that evaluate the results of classification.

        :param y_true: The actual class vector/matrix (assuming one-hot encoding in the latter case).
        :type y_true: np.ndarray
        :param y_pred: The predicted class vector/matrix (assuming one-hot encoding in the latter case).
        :type y_pred: np.ndarray
        :param classes: Vector of unique class values. Required for single output classification accuracy calculation.
        :type classes: None | np.ndarray
        :cvar _metric_modes: Map for defining which metric is suitable for a certain target type.
        :type _metric_modes: dict[Callable[[], float], str]

        :return: None
        :rtype: None
        """

        self.true = y_true
        self.predicted = y_pred
        self.classes = classes

        if self.true.shape != self.predicted.shape: raise ValueError("Target shape mismatch!")

        if len(self.true.shape) > 1: self.class_form = "one-hot"
        else: self.class_form = "normal"

    def accuracy_oh(self) -> float:
        """
        Accuracy with one-hot encoded target.
        :return: Classification accuracy
        :rtype: float
        """

        return accuracy_score(np.argmax(self.true, axis=1), np.argmax(self.predicted, axis=1))

    def accuracy_reg(self) -> float:
        """
        Accuracy of classification calculated from single output (regression).
        :return: Classification accuracy
        :rtype: float
        """
        if self.classes is None: raise ValueError("A vector of unique class values must be given!")

        y_true_conv = self.true.reshape(1, -1)
        y_pred_conv = self.predicted.reshape(1, -1)
        classes_conv = self.classes.reshape(-1, 1)

        return np.mean(np.argmin(np.abs(classes_conv - y_true_conv), axis=0) == np.argmin(np.abs(classes_conv - y_pred_conv), axis=0))

    def root_mean_squared_oh(self) -> float:
        """
        Root-Mean Squared error with one-hot encoded target.
        :return: RMS error
        :rtype: float
        """

        return np.sqrt(np.mean(np.square(np.argmax(self.predicted, axis=1) - np.argmax(self.true, axis=1))))

    def scaled_root_mean_squared_oh(self) -> float:
        """
        Scaled Root-Mean Squared error with one-hot encoded target. Used with target vectors scaled between 0.1 and 0.9.
        :return: SRMS error
        :rtype: float
        """

        return np.sqrt(np.mean(np.square(np.argmax(self.predicted, axis=1) - np.argmax(self.true, axis=1))) / ((0.8 ** 2) / 2))

    def root_mean_squared(self) -> float:
        """
        Root-Mean Squared error.
        :return: RMS error
        :rtype: float
        """

        return np.sqrt(np.mean(np.square(self.predicted - self.true)))

    def scaled_root_mean_squared(self) -> float:
        """
        Scaled Root-Mean Squared error. Used with target vectors scaled between 0.1 and 0.9.
        :return: SRMS error
        :rtype: float
        """

        return np.sqrt(np.mean(np.square(self.predicted - self.true)) / ((0.8 ** 2) / 2))

    def calculate(self, metrics: list[Callable[[], float]] | str = "all") -> dict[str, float]:
        """
        Calculate the given metrics based on the target labels. The functions are called based on the class vector shape.

        :param metrics: List of functions to call. Alternatively, "all" to calculate all metrics. Default value is "all".
        :return: Dictionary with metric names as keys and scores as values.
        :rtype: dict[str, float]
        """

        functions = None
        if type(metrics) == str:
            functions = [f for f in self._metric_modes.keys()]
        else:
            functions = metrics

        results = {}

        for f in functions:
            if self._metric_modes[f] == self.class_form:
                results[f.__name__] = f()

        return results