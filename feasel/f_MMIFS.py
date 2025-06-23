import sys
import os
sys.path.insert(0, os.path.abspath('feasel/helpers'))

from feasel.helpers.feasel_frame import FeaselBase

import numpy as np
import time

class ModifiedMutualInfo(FeaselBase):
    @property
    def _flags(self) -> dict[str, bool]:
        return {
            "multilabel": False,
            "multiclass": True,
            "supervised": False,
            "discrete": False,
            "missing_values": False,
            "requires_selected": True
        }

    @property
    def _measures_used(self) -> set[str]:
        return {"MIit", "MIij"}

    @property
    def _name(self) -> str:
        return "MMIFS"

    def __init__(self, dataset: np.ndarray, target: np.ndarray, selected: set[int] = None, n_features: int = 1, beta: float = 0.5, verbose: int = 0):
        """
        Implements the modified Mutual Information Feature Selection found in
        "Mutual information-based feature selection for intrusion detection systems" by Amiri et al.
        https://doi.org/10.1016/j.jnca.2011.01.002

        :param dataset: Dataset of size (n_samples, n_features).
        :type dataset: np.ndarray
        :param target: Target vector of size (n_samples,).
        :type target: np.ndarray
        :param selected: Already selected feature index set of size (n_selected,). Default value is None.
        :type selected: set[int]
        :param n_features: Number of features to select. Default value is 1.
        :type n_features: int
        :param beta: Weight of feature redundancy. Default value is 0.5.
        :type beta: float
        :param verbose: Verbosity. 0: no output; 1: prints execution time, selected feature and metric; 2: prints every step. Recommend turning off parallel execution when verbose is 2. Note that increased verbosity affects execution time. Default value is 0.
        :type verbose: int
        :cvar _flags: Dictionary of flags describing the capabilities of the algorithm.
        :type _flags: dict[str, bool]
        :cvar _measures_used: Set of measure names used in the algorithm.
        :type _measures_used: set[str]

        :return: None
        :rtype: None
        """
        super().__init__(dataset, target, selected, n_features, verbose)
        self.beta = beta

    def transform(self) -> tuple[set[int], list[dict[int, float]], float]:
        """
        Applies the algorithm.

        :return: Selected feature index or indices, measure values for all candidate features, execution time; tuple of size (3,).
        :rtype: tuple[set[int], list[dict[int, float]], float]
        """
        if self.measures is None: raise Exception("Missing measures dictionary; fit must be called before transform!")
        start_time = time.time()

        if self.verbose > 0: print(f"{self._name} measures: {self._measures_used}")

        all_candidates: list[dict[int, float]] = []
        if len(self.selected) == 0:
            if self.verbose > 1: print(f"{self._name}: Selected features vector is empty, appending feature {np.argmax(self.measures['MIit'])}..")

            self.selected.add(np.argmax(self.measures["MIit"]))
            all_candidates.append({np.argmax(self.measures["MIit"]): np.max(self.measures["MIit"])})

        while len(self.selected) < self.n_features:
            if self.verbose > 1: print(f"{self._name}: Selecting additional features: {self.n_features - len(self.selected)} remaining..")
            candidates = np.array([i for i in range(self.data.shape[1]) if i not in self.selected]).reshape(-1, 1)

            indices = np.concatenate([np.where(list(self.selected)<candidates, list(self.selected), candidates).flatten("F").reshape(-1, 1),
                       np.where(list(self.selected)<candidates, candidates, list(self.selected)).flatten("F").reshape(-1, 1)], axis=-1)

            measure = self.measures["MIij"][indices[:, 0], indices[:, 1]]
            measure_candidate_list = indices[np.isin(indices, candidates)]
            measure_same_candidate = np.argwhere(candidates.reshape(-1, 1) == measure_candidate_list)[:, 1].reshape(-1, len(self.selected))

            candidates_MIij = {candidates[i][0]: np.sum(measure[measure_same_candidate[i]]) for i in range(len(candidates))}
            feature_score = {i: self.measures["MIit"][i] - (self.beta / len(self.selected)) * candidates_MIij[i] for i in candidates_MIij.keys()}

            if self.verbose > 1: [print(f"{self._name}: Candidate feature {i} score is: {feature_score[i]}") for i in feature_score.keys()]

            self.selected.add(max(feature_score, key=feature_score.get))
            all_candidates.append(feature_score)
            if self.verbose > 1: print(f"{self._name}: Selected feature {max(feature_score, key=feature_score.get)}!")

        end_time = time.time()
        if self.verbose > 0: print(f"{self._name} transform time: {end_time - start_time}")
        if self.verbose > 0: print(f"{self._name} selected feature(s): {self.selected}")
        return self.selected, all_candidates, end_time - start_time