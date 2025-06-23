import sys
import os

sys.path.insert(0, os.path.abspath('feasel/helpers'))

from feasel.helpers.feasel_frame import FeaselBase

import numpy as np
import time

from pyitlib.discrete_random_variable import entropy
from pyitlib.discrete_random_variable import information_mutual

class FastCorriPFS(FeaselBase):
    @property
    def _flags(self) -> dict[str, bool]:
        return {
            "multilabel": False,
            "multiclass": True,
            "supervised": False,
            "discrete": True,
            "missing_values": False,
            "requires_selected": False
        }

    @property
    def _measures_used(self) -> set[str]:
        return {"SUit"}

    @property
    def _name(self) -> str:
        return "FCBFiP"

    def _calculate_SU(self, X: np.ndarray, Y: np.ndarray) -> float:
        return 2 * (information_mutual(X, Y, base = 10) / (entropy(X, base = 10) + entropy(Y, base = 10)))

    def _is_prime(self, a: int) -> bool:
        all_num = np.arange(2, a, dtype=int)
        return np.all(a % all_num)

    def __init__(self, dataset: np.ndarray, target: np.ndarray, n_pieces: int|None = None, n_features: int|None = None, verbose: int = 0):
        """
        Implements the Fast Correlation Based Feature Selection in Pieces found in
        "Intelligent IoT Traffic Classification Using Novel Search Strategy for Fast-Based-Correlation Feature Selection in Industrial Environments" by Egea et al.
        https://doi.org/10.1109/JIOT.2017.2787959

        :param dataset: Dataset of size (n_samples, n_features).
        :type dataset: np.ndarray
        :param target: Target vector of size (n_samples,).
        :type target: np.ndarray
        :param n_pieces: Number of pieces to divide the feature space into. If None or unsuitable for the given dataset, this value is automatically set to the lowest possible during fit. Default is None.
        :type n_pieces: int
        :param n_features: Number of features to select. If None, returns with a feature order. Default value is 1.
        :type n_features: int
        :param verbose: Verbosity. 0: no output; 1: prints execution time, selected feature and metric; 2: prints every step. Recommend turning off parallel execution when verbose is 2. Note that increased verbosity affects execution time. Default value is 0.
        :type verbose: int
        :cvar _flags: Dictionary of flags describing the capabilities of the algorithm.
        :type _flags: dict[str, bool]
        :cvar _measures_used: Set of measure names used in the algorithm.
        :type _measures_used: set[str]

        :return: None
        :rtype: None
        """
        if n_features is None:
            super().__init__(dataset, target, set(), dataset.shape[1], verbose)
        else:
            super().__init__(dataset, target, set(), n_features, verbose)

        self.n_pieces = n_pieces
        self.SUit_sorted = None
        self.row_modifier = None
        self.score = None

        self.removed = None

    def fit(self, measures: dict[str, np.ndarray]) -> float:
        """
        Getter function for measure matrices used in the algorithm. Checks the validity of the n_pieces parameter and adjusts if needed.

        :param measures: Dictionary of measure matrices used in the algorithm.
        :type measures: dict[str, np.ndarray]

        :return: fit time
        :rtype: float
        """
        if self.data.shape[1] < 4:
            if self.verbose > 0: print(f"{self._name}: Dataset feature count is below 4, skipping fit..")
            return 0

        start_time = time.time()

        self.measures = {k: measures[k] for k in self._measures_used}

        self.SUit_sorted = np.argsort(self.measures["SUit"])[::-1]

        adjust_flag = self.n_pieces is not None
        if adjust_flag: adjust_flag = self.data.shape[1] % self.n_pieces == 0 and self.n_pieces > 1

        if not adjust_flag:
            nums = np.arange(2, self.data.shape[1])
            divisors = np.where(self.data.shape[1] % nums == 0)

            if len(divisors[0]) == 0:
                if self.verbose > 1: print(f"{self._name}: Dataset feature count is a prime number, deleting lowest SU feature..")
                self.removed = self.SUit_sorted[-1]
                self.SUit_sorted = self.SUit_sorted[:-1]

                nums = nums[:-1]
                divisors = np.where(self.SUit_sorted.shape[0] % nums == 0)

            self.n_pieces = nums[np.min(divisors)]
            if self.verbose > 0: print(f"{self._name}: Setting n_pieces to {self.n_pieces}")

        self.row_modifier = self.SUit_sorted.shape[0] // self.n_pieces

        end_time = time.time()
        return end_time - start_time

    def transform(self) -> tuple[set[int], np.ndarray[int], float]:
        """
        Applies the algorithm.

        :return: Selected feature index or indices, ordered selection, execution time; tuple of size (3,).
        :rtype: tuple[set[int], np.ndarray[int], float]
        """
        if self.data.shape[1] < 4:
            if self.verbose > 0: print(f"{self._name}: Dataset feature count is below 4, skipping transform..")
            return self.selected, np.array([]), 0

        if self.measures is None: raise Exception("Missing measures dictionary; fit must be called before transform!")
        start_time = time.time()

        if self.verbose > 0: print(f"{self._name} measures: {self._measures_used}")

        reshaped_X = np.reshape(self.data[:, self.SUit_sorted].T, (self.n_pieces, self.row_modifier * self.data.shape[0])).T

        SU_vicinity = np.array([])
        for f in range(reshaped_X.shape[1]):
            X_vicinity = np.reshape(reshaped_X[:, f], (self.row_modifier, reshaped_X.shape[0] // self.row_modifier)).T

            SUij = np.zeros((X_vicinity.shape[1], X_vicinity.shape[1]))
            for i in range(X_vicinity.shape[1]):
                for j in range(i + 1, X_vicinity.shape[1]):
                    SUij[i, j] = self._calculate_SU(X_vicinity[:, i], X_vicinity[:, j])
                    SUij[j, i] = SUij[i, j]

            SU_vicinity = np.append(SU_vicinity, 1 / (X_vicinity.shape[1] - 1) * np.sum(SUij, axis = 1))

        SU_vicinity_ordered = np.argsort(SU_vicinity)
        if self.removed is not None: np.add.at(SU_vicinity_ordered, np.where(self.removed <= SU_vicinity_ordered), 1)

        self.score = np.empty((self.data.shape[1],))
        self.score[:] = np.nan

        for i in self.SUit_sorted:
            self.score[i] = np.argwhere(i == self.SUit_sorted) + np.argwhere(i == SU_vicinity_ordered)

        selected_features = np.argsort(self.score)

        if selected_features.shape[0] > self.n_features:
            selected_features = selected_features[:self.n_features]

        [self.selected.add(i) for i in selected_features]

        end_time = time.time()
        if self.verbose > 0: print(f"{self._name} transform time: {end_time - start_time}")
        if self.verbose > 0: print(f"{self._name} selected feature(s) sorted: {self.score}")
        return self.selected, selected_features, end_time - start_time