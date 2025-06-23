import abc
import numpy as np

from utils.metrics import Metrics

class EvalBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        """
        Abstract class for learning algorithms in the Adaptive, Hybrid Feature Selection framework.
        """

        pass

    @abc.abstractmethod
    def fit(self, **kwargs) -> None:
        """
        Abstract method for fitting the learning algorithm.

        :param kwargs: Keyword arguments specific to individual evaluators.
        """

        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract function for predicting the target vector on the given data.

        :param X: Data to predict the target vector from.
        :type X: np.ndarray
        :return: The predicted label vector.
        :rtype: np.ndarray
        """

        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> Metrics:
        """
        Performs a prediction, then compares with the true class labels. Returns a Metrics object.

        :param X: Data to predict the target vector from.
        :type X: np.ndarray
        :param y: Ground truth target vector.
        :type y: np.ndarray
        :return: Metrics object containing the predicted and true class labels.
        :rtype: Metrics
        """

        y_pred = self.predict(X)

        classes = None
        if len(y.shape) > 1:
            classes = np.unique(np.argmax(y, axis=1))
        else:
            classes = np.unique(y)

        return Metrics(y, y_pred, classes)