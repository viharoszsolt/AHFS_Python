import sys
import os
import warnings

sys.path.insert(0, os.path.abspath('feasel/helpers'))

from feasel.helpers.feasel_frame import FeaselBase

import numpy as np
import time

class FastCorrFS(FeaselBase):
    @property
    def _flags(self) -> dict[str, bool]:
        return {
            "multilabel": False,
            "multiclass": True,
            "supervised": False,
            "discrete": False,
            "missing_values": False,
            "requires_selected": False
        }

    @property
    def _measures_used(self) -> set[str]:
        return {"SUit", "SUij"}

    @property
    def _name(self) -> str:
        return "FCBF"

    def __init__(self, dataset: np.ndarray, target: np.ndarray, max_features: int|None = None, threshold: float = 0.01, verbose: int = 0):
        """
        Implements the Fast Correlation Based Feature Selection found in
        "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution" by Yu et al.
        https://www.researchgate.net/publication/221345776_Feature_Selection_for_High-Dimensional_Data_A_Fast_Correlation-Based_Filter_Solution

        :param dataset: Dataset of size (n_samples, n_features).
        :type dataset: np.ndarray
        :param target: Target vector of size (n_samples,).
        :type target: np.ndarray
        :param max_features: Maximum number of features to select. If None, returns with all suitable candidates. Default value is None.
        :type max_features: int
        :param threshold: Threshold for initial filtering. Default value is 0.01.
        :type threshold: float
        :param verbose: Verbosity. 0: no output; 1: prints execution time, selected feature and metric; 2: prints every step. Recommend turning off parallel execution when verbose is 2. Note that increased verbosity affects execution time. Default value is 0.
        :type verbose: int
        :cvar _flags: Dictionary of flags describing the capabilities of the algorithm.
        :type _flags: dict[str, bool]
        :cvar _measures_used: Set of measure names used in the algorithm.
        :type _measures_used: set[str]

        :return: None
        :rtype: None
        """
        if max_features is None:
            super().__init__(dataset, target, set(), dataset.shape[1], verbose)
        else:
            super().__init__(dataset, target, set(), max_features, verbose)

        self.threshold = threshold

    def transform(self) -> tuple[set[int], np.ndarray[int], float]:
        """
        Applies the algorithm.

        :return: Selected feature index or indices, ordered selection, execution time; tuple of size (3,).
        :rtype: tuple[set[int], np.ndarray[int], float]
        """
        if self.measures is None: raise Exception("Missing measures dictionary; fit must be called before transform!")
        start_time = time.time()

        if self.verbose > 0: print(f"{self._name} measures: {self._measures_used}")

        th_filter = np.where(self.measures["SUit"] > self.threshold)[0]
        SUit_sorted = th_filter[np.argsort(self.measures["SUit"][th_filter])][::-1]

        if self.verbose > 1: print(f"{self._name}: Post-filter feature order: {SUit_sorted}")
        if SUit_sorted.shape[0] == 0:
            warnings.warn(f"{self._name}: Threshold too high, no eligible candidates remaining!")
            return self.selected, {}, 0

        i = 0

        while i < len(SUit_sorted) - 1:
            indices = np.concatenate([np.where(SUit_sorted[i] < SUit_sorted[i + 1:], SUit_sorted[i], SUit_sorted[i + 1:]).reshape(-1, 1),
                                            np.where(SUit_sorted[i] < SUit_sorted[i + 1:], SUit_sorted[i + 1:], SUit_sorted[i]).reshape(-1, 1)],
                                            axis=-1)

            try:
                to_remove = np.argwhere(self.measures["SUij"][indices[:, 0], indices[:, 1]] > self.measures["SUit"][SUit_sorted[i + 1:]]) + (i + 1)
            except IndexError:
                to_remove = []

            if self.verbose > 1: print(f"{self._name}: Predominant feature is {SUit_sorted[i]}, removing feature(s) {SUit_sorted[to_remove].reshape(1, -1)}..")

            SUit_sorted = np.delete(SUit_sorted, to_remove)
            i += 1

        if SUit_sorted.shape[0] > self.n_features:
            SUit_sorted = SUit_sorted[:self.n_features]

        [self.selected.add(i) for i in SUit_sorted]

        end_time = time.time()
        if self.verbose > 0: print(f"{self._name} transform time: {end_time - start_time}")
        if self.verbose > 0: print(f"{self._name} selected feature(s) sorted: {SUit_sorted}")
        return self.selected, SUit_sorted, end_time - start_time