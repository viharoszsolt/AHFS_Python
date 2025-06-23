import sys
import os

sys.path.insert(0, os.path.abspath('feasel/helpers'))

from feasel.helpers.feasel_frame import FeaselBase

import numpy as np
import time

class DistanceFS(FeaselBase):
    @property
    def _flags(self) -> dict[str, bool]:
        return {
            "multilabel": False,
            "multiclass": True,
            "supervised": True,
            "discrete": False,
            "missing_values": False,
            "requires_selected": True
        }

    @property
    def _measures_used(self) -> None:
        return None

    @property
    def _name(self) -> str:
        return "ORIG"

    def __init__(self, dataset: np.ndarray, target: np.ndarray, selected: set[int], n_features: int = 1, metric: str = "weighted euclidean", verbose: int = 0):
        """
        Implements the Distance Based Feature Selection found in "Adaptive Hybrid Feature Selection (AHFS)" by Viharos et al.
        https://doi.org/10.1016/j.patcog.2021.107932

        :param dataset: Dataset of size (n_samples, n_features).
        :type dataset: np.ndarray
        :param target: Target vector of size (n_samples,).
        :type target: np.ndarray
        :param selected: Already selected set of feature indices of size (n_selected,).
        :type selected: set[int]
        :param n_features: Maximum number of features to select. Default value is 1.
        :type n_features: int
        :param metric: The distance metric to use. Default value is "weighted euclidean".
        :type metric: str
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

        if metric in ["euclidean", "weighted euclidean"]:
            self.metric = metric
        else:
            raise NotImplementedError(f"Metric {metric} is not implemented!")

    def transform(self) -> tuple[set[int], float]:
        """
        Applies the algorithm.

        :return: Selected feature index or indices, execution time; tuple of size (2,).
        :rtype: tuple[set[int], float]
        """
        start_time = time.time()

        if self.verbose > 0: print(f"{self._name} measures: {self._measures_used}")

        feature_combinations = np.arange(self.data.shape[1])
        feature_combinations = np.delete(feature_combinations, list(self.selected), axis=0).reshape(-1, 1).astype(int)
        for i in self.selected:
            feature_combinations = np.insert(feature_combinations, 0, i, axis=1)

        if self.n_features > 1:
            feature_combinations = feature_combinations[np.isin(feature_combinations, list(self.selected)).sum(axis=1)
                                                        == len(self.selected)]

        labels = np.unique(self.target)
        combination_score = np.full(feature_combinations.shape[0], np.nan)

        for i, f in enumerate(feature_combinations):
            centroids = np.full((labels.shape[0], self.n_features), np.nan)
            distances = np.full(labels.shape[0], np.nan)
            centroids_distances = np.full(labels.shape[0], np.nan)

            class_sizes = []
            M = np.array([])
            for l in range(labels.shape[0]):
                extracted_samples = self.data[self.target == labels[l]][:, f]
                class_sizes.append(len(extracted_samples))

                centroids[l] = np.mean(extracted_samples, axis=0)

                if self.metric == "euclidean":
                    distances[l] = np.sqrt(np.square(centroids[l] - extracted_samples).sum(axis=1)).mean()
                elif self.metric == "weighted euclidean":
                    M = np.append((extracted_samples - centroids[l]).reshape(1, -1), M)

            # Calculate the center of the centroids, and the distance between the centroids and their center
            c_centroid = np.mean(self.data[:, f], axis=0)

            if self.metric == "euclidean":
                centroids_distances = np.sqrt(np.square(c_centroid - centroids).sum(axis=1)).mean()
                swtrace = np.mean(distances)
            elif self.metric == "weighted euclidean":
                centroids_distances = np.sum(np.sum((c_centroid - centroids) ** 2, axis=1) * np.array(class_sizes)) / np.sum(class_sizes)
                swtrace = np.sum(np.nansum(M.reshape(-1, 1) ** 2, axis=1)) / self.data.shape[0]

            combination_score[i] = swtrace / centroids_distances

        best_set = feature_combinations[np.nanargmin(combination_score)]
        candidate = best_set[np.isin(best_set, list(self.selected), invert = True)]
        {self.selected.add(i) for i in candidate}

        end_time = time.time()
        if self.verbose > 0: print(f"{self._name} transform time: {end_time - start_time}")
        if self.verbose > 0: print(f"{self._name} selected feature(s): {self.selected}")
        return self.selected, end_time - start_time