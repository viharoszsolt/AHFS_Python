import abc
import numpy as np

class FeaselBase(abc.ABC):
    @property
    @abc.abstractmethod
    def _flags(self) -> dict[str, bool]:
        return {
            "multilabel": False,
            "multiclass": False,
            "supervised": False,
            "discrete": False,
            "sparse": False,
            "missing_values": False,
            "requires_selected": False
        }

    @property
    @abc.abstractmethod
    def _measures_used(self) -> set[str]:
        pass

    @property
    @abc.abstractmethod
    def _name(self) -> str:
        return "UNDEFINED"

    def __init__(self, dataset: np.ndarray, target: np.ndarray, selected: set[int] = None, n_features: int = 1, verbose: int = 0):
        """
        Abstract base class for all feature selection algorithms within the Adaptive, Hybrid Feature Selection framework.

        :param dataset: Dataset of size (n_samples, n_features).
        :type dataset: np.ndarray
        :param target: Target vector of size (n_samples,).
        :type target: np.ndarray
        :param selected: Already selected feature index set of size (n_selected,). Default value is None.
        :type selected: set[int]
        :param n_features: Number of features to select. Default value is 1.
        :type n_features: int
        :param verbose: Verbosity. 0: no output; 1: prints execution time, selected feature and metric; 2: prints every step. Recommend turning off parallel execution when verbose is 2. Default value is 0.
        :type verbose: int
        :cvar _flags: Dictionary of flags describing the capabilities of the algorithm.
        :type _flags: dict[str, bool]
        :cvar _measures_used: Set of measure names used in the algorithm.
        :type _measures_used: set[str]
        :cvar _name: Name of the algorithm.
        :type _name: str

        :return: None
        :rtype: None
        """

        self.data = dataset
        self.target = target
        self.selected = selected
        self.n_features = n_features
        self.verbose = verbose

        self.measures = None

        if self._flags["multiclass"] and len(np.unique(self.target)) < 2: raise ValueError(f"{self._name} is only for binary classification.")
        if self._flags["supervised"] and len(np.unique(self.target)) is None: raise ValueError(f"{self._name} requires a target vector.")
        if self._flags["missing_values"] and np.isnan(self.data).any(): raise ValueError(f"{self._name} does not support missing values.")
        if self._flags["requires_selected"] and self.selected is None: raise ValueError(f"{self._name} requires a set of selected features.")

    def fit(self, measures: dict[str, np.ndarray]) -> None:
        """
        Getter function for measure matrices used in the algorithm.

        :param measures: Dictionary of measure matrices used in the algorithm.
        :type measures: dict[str, np.ndarray]

        :return: None
        :rtype: None
        """

        if measures is not None and self._measures_used is not None:
            self.measures = {k: measures[k] for k in self._measures_used}

    @abc.abstractmethod
    def transform(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Abstract transform method.

        :return: Selected feature index or indices, measure values for all features, feature order, execution time; tuple of size (4,).
        :rtype: tuple
        """

        pass
