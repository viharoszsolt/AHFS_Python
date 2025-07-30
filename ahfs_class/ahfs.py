from datetime import datetime
import time
import random
from typing import Callable

import joblib
import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import r_regression
from joblib import Parallel, delayed

from feasel import f_FFSA
from feasel import f_MMIFS
from feasel import f_NMIFS
from feasel import f_LCFS
from feasel import f_FCBF
from feasel import f_mFCBF
from feasel import f_FCBFiP
from feasel import f_ORIG
from feasel import f_mRMR

from evaluators.e_AgiLM import AgiLM, CONST_GPU_THRESHOLD
from evaluators.e_AgiLM import sigmoid

from utils.preprocessing import discretize_X_y

class AHFS:
    def __init__(self, k: int, data_bin: int = 5, target_bin: int = 2, save_precomp: bool = True, save_precomp_path: str | None = None, load_precomp_path: str | None = None, is_in_pipeline: bool = False, verbose: int = 1):
        """
        Implements the Adaptive Hybrid Feature Selection (AHFS) algorithm by Viharos et al.
        https://doi.org/10.1016/j.patcog.2021.107932

        :param k: Number of features to select.
        :type k: int
        :param data_bin: How many bins to discretize the dataset into, excluding the target variable. If 0, no discretization is performed. Default value is 5.
        :type data_bin: int
        :param target_bin: How many bins to discretize the target into. Effectively sets the number of classes. If 0, no discretization is performed. Default value is 2.
        :type target_bin: int
        :param save_precomp: Whether to save all basic measures computed during the precomputing phase in a binary .npy file. Path is set by save_precomp_path. Default value is True.
        :type save_precomp: bool
        :param save_precomp_path: If save_precomp is True, sets the path for saving the precomputed basic measures. If this variable's value is None, saves into the current directory with filename format "measures_{time.time_ns()}.npy". Default value is None.
        :type save_precomp_path: str | None
        :param load_precomp_path: Path to load the precomputed basic measures from, skipping the precomputing phase. File must be a binary numpy file. If None, precomputing is not skipped. Default value is None.
        :type load_precomp_path: str | None
        :param is_in_pipeline: Set to True if the algorithm is intended to be part of a scikit-learn pipeline. Changes the return value of transform() to X and y. Default value is False.
        :type is_in_pipeline: bool
        :param verbose: Controls global verbosity, including feature selection measures. 0: no output, 1: feature selection measure execution time, selected feature and metric, and basic algorithm steps are printed, 2: all steps are printed. Default value is 1.
        :type verbose: int
        """

        self.k = k
        self.data_bin = data_bin
        self.target_bin = target_bin
        self.save_precomp = save_precomp
        self.save_precomp_path = save_precomp_path
        self.load_precomp_path = load_precomp_path
        self.is_in_pipeline = is_in_pipeline
        self.verbose = verbose

        self.X_n = None
        self.y_n = None

        self.X_d = None
        self.y_d = None

        self.selected = set({})
        self.selected_sorted = []
        self.iter_selected = None

        self.loss = []
        self.accuracy = []

        self.measures_times = []
        self.nn_times = []

        self.measures = {}

    def entropies_wrapper(self, index: tuple[int, int] | tuple[int, int, int]) -> list[float, tuple]:
        if len(index) > 2:
            return [drv.entropy_joint([self.X_d[:, index[0]], self.X_d[:, index[1]], self.y_d]), index]
        elif index[1] == -1:
            return [drv.entropy(self.X_d[:, index[0]]), index]
        elif index[1] != 0:
            return [drv.entropy_joint([self.X_d[:, index[0]], self.X_d[:, index[1]]]), index]
        elif index[1] == 0:
            return [drv.entropy_joint([self.X_d[:, index[0]], self.y_d]), index]
        else:
            return NotImplementedError(f"Undefined behavior with arg {index}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Placeholder function to ensure compatibility with scikit-learn interface.
        """

        pass

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[list[int], list[float], list[float], dict[str: int], list[float], float] | tuple[np.ndarray, np.ndarray]:
        """
        Applies the Adaptive Hybrid Feature Selection algorithm on the dataset.

        :param X: Numpy array holding the data.
        :type X: np.ndarray
        :param y: Target vector.
        :type y: np.ndarray
        :return: If is_in_pipeline was set to False: A list of the selected features, loss of the selected feature set, accuracy of the selected feature set, features selected per iteration. If is_in_pipeline was set to True: The original dataset containing only the selected features, target variable.
        """
        transform_time_start = time.time()

        if self.verbose > 0: print(f"{datetime.now()}: Starting AHFS..")

        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        self.X_n = scaler.fit_transform(X)
        self.y_n = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1,)

        if self.X_d is None or self.y_d is None:
            self.X_d, self.y_d = discretize_X_y(self.X_n, self.y_n, X_bins = self.data_bin, y_bins = self.target_bin)

        if self.data_bin == 0: self.X_d = X.astype(int)
        if self.target_bin == 0: self.y_d = y.astype(int)

        if self.load_precomp_path is not None:
            self.measures = np.load(self.load_precomp_path, allow_pickle = True).item()
        else:
            MIij = np.zeros((X.shape[1], X.shape[1]))
            SUij = np.zeros((X.shape[1], X.shape[1]))
            CORRij = np.zeros((X.shape[1], X.shape[1]))
            JMIijt = np.zeros((X.shape[1], X.shape[1]))
            SUit = np.zeros((X.shape[1],))
            MIit = np.zeros((X.shape[1],))
            Ei = np.zeros((X.shape[1],))
            Et = np.zeros((1,))
            Eit = np.zeros((X.shape[1],))
            JEij = np.zeros((X.shape[1], X.shape[1]))
            JEijt = np.zeros((X.shape[1], X.shape[1]))

            pair_args = [(i, j) for i in range(X.shape[1]) for j in range(i + 1, X.shape[1])]
            pair_args += [(i, 0) for i in range(X.shape[1])]

        if self.verbose > 0: print(f"{datetime.now()}: Starting process pool..")
        with joblib.parallel_config(n_jobs=-1):
            if self.load_precomp_path is None:
                precomp_time = time.time()

                if self.verbose > 0: print(f"{datetime.now()}: Starting precomputing..")
                if self.verbose > 0: print(f"{datetime.now()}: Part 1..")

                single_args = [(i, -1) for i in range(X.shape[1])]
                triple_args = [(i, j, 0) for i, j in pair_args]

                entropies_args = single_args + pair_args + triple_args

                j = 0
                for result in Parallel(return_as="generator_unordered")(delayed(self.entropies_wrapper)(i) for i in entropies_args):
                    j += 1
                    if j % 100 == 0:
                        if self.verbose > 0: print(f"{datetime.now()}: Part 1 progress: {j}/{len(entropies_args)}")

                    if len(result[1]) == 2:
                        if result[1][1] == 0:
                            Eit[result[1][0]] = result[0]
                        elif result[1][1] != -1:
                            JEij[result[1][0], result[1][1]] = result[0]
                        elif result[1][1] == -1:
                            Ei[result[1][0]] = result[0]
                        else:
                            raise NotImplementedError(f"Undefined behavior with arg {result[1]}")

                    elif len(result[1]) == 3:
                        JEijt[result[1][0], result[1][1]] = result[0]

                    else:
                        raise NotImplementedError(f"Undefined behavior with arg {result[1]}")

                Et = drv.entropy(self.y_d)

                if self.verbose > 0: print(f"{datetime.now()}: Part 2..")

                base_measures = {
                    "Ei": Ei,
                    "Et": Et,
                    "Eit": Eit,
                    "JEij": JEij,
                    "JEijt": JEijt
                }

                MIij = base_measures["Ei"][:, None] + base_measures["Ei"][None, :] - base_measures["JEij"]
                CORRij = np.abs(np.corrcoef(self.X_d, rowvar=False))
                JMIijt = base_measures["JEij"] + base_measures["Et"] - base_measures["JEijt"]
                MIit = base_measures["Ei"] + base_measures["Et"] - base_measures["Eit"]

                CORRit = abs(r_regression(self.X_d, self.y_d, force_finite = True))

                if self.verbose > 0: print(f"{datetime.now()}: Part 3..")

                su_measures = {
                    "MIij": MIij,
                    "MIit": MIit,
                    "Ei": Ei,
                    "Et": Et
                }

                SUij = 2 * su_measures["MIij"] / (su_measures["Ei"][:, None] + su_measures["Ei"][None, :])
                SUit = 2 * su_measures["MIit"] / (su_measures["Ei"] + su_measures["Et"])

                self.measures["MIij"] = MIij
                self.measures["SUij"] = SUij
                self.measures["CORRij"] = CORRij
                self.measures["JMIijt"] = JMIijt
                self.measures["SUit"] = SUit
                self.measures["MIit"] = MIit
                self.measures["CORRit"] = CORRit
                self.measures["Ei"] = Ei

                if self.save_precomp:
                    if self.save_precomp_path is None:
                        np.save(f"measures_{time.time_ns()}", np.array(self.measures))
                    else:
                        np.save(self.save_precomp_path, np.array(self.measures))

                precomp_time = time.time() - precomp_time
                if self.verbose > 0: print(f"{datetime.now()}: Precomputing time:", precomp_time)

            if self.verbose > 0: print(f"{datetime.now()}: Starting measures..")

            ### FCBF ###
            fcbf = f_FCBF.FastCorrFS(dataset = self.X_d, target = self.y_d, threshold = 0, verbose = self.verbose)
            fcbf.fit(self.measures)

            ### FCBF# ###
            fcbf2 = f_mFCBF.ModifiedFastCorrFS(dataset = self.X_d, target = self.y_d, threshold = 0, verbose = self.verbose)
            fcbf2.fit(self.measures)

            ### FCBFiP ###
            fcbfip = f_FCBFiP.FastCorriPFS(dataset = self.X_d, target = self.y_d, verbose = self.verbose)
            fcbfip.fit(self.measures)

            S_fcbf, S_ordered_fcbf, fcbf_time = None, None, None
            S_fcbf2, S_ordered_fcbf2, fcbf2_time = None, None, None
            S_fcbfip, S_ordered_fcbfip, fcbfip_time = None, None, None

            fcbf_measures = [fcbf, fcbf2, fcbfip]
            for i, result in enumerate(Parallel(return_as="generator")(delayed(cl.transform)() for cl in fcbf_measures)):
                match i:
                    case 0:
                        S_fcbf, S_ordered_fcbf, fcbf_time = result
                        if self.verbose > 0: print(f"{datetime.now()}: FCBF done")
                    case 1:
                        S_fcbf2, S_ordered_fcbf2, fcbf2_time = result
                        if self.verbose > 0: print(f"{datetime.now()}: FCBF# done")
                    case 2:
                        S_fcbfip, S_ordered_fcbfip, fcbfip_time = result
                        if self.verbose > 0: print(f"{datetime.now()}: FCBFiP done")

            while len(self.selected) < self.k:
                measures_start = time.time()
                to_evaluate = []

                ffsa = f_FFSA.ForwardFeatureSel(dataset = self.X_d, target = self.y_d, selected = self.selected, n_features = len(self.selected) + 1, verbose = self.verbose)
                ffsa.fit(self.measures)

                mmifs = f_MMIFS.ModifiedMutualInfo(dataset = self.X_d, target = self.y_d, selected = self.selected, n_features = len(self.selected) + 1, verbose = self.verbose)
                mmifs.fit(self.measures)

                nmifs = f_NMIFS.NormalizedMutualInfo(dataset=self.X_d, target=self.y_d, selected=self.selected, n_features=len(self.selected) + 1, verbose=self.verbose)
                nmifs.fit(self.measures)

                lcfs = f_LCFS.LinearCorrelation(dataset = self.X_d, target = self.y_d, selected = self.selected, n_features = len(self.selected) + 1, verbose = self.verbose)
                lcfs.fit(self.measures)

                mrmr = f_mRMR.MinRedMaxRel(dataset = self.X_d, target = self.y_d, selected = self.selected, n_features = len(self.selected) + 1, verbose = self.verbose)
                mrmr.fit(self.measures)

                mrmr2 = f_mRMR.MinRedMaxRel(method="modified", dataset = self.X_d, target = self.y_d, selected = self.selected, n_features = len(self.selected) + 1, verbose = self.verbose)
                mrmr2.fit(self.measures)

                orig = f_ORIG.DistanceFS(dataset = self.X_n, target = self.y_d, selected = self.selected, n_features = len(self.selected) + 1, verbose = self.verbose)
                orig.fit(self.measures)

                S_ffsa, ffsa_time = None, None
                S_mmifs, mmifs_time = None, None
                S_nmifs, nmifs_time = None, None
                S_lcfs, lcfs_time = None, None
                S_mrmr, mrmr_time = None, None
                S_mrmr2, mrmr2_time = None, None
                S_orig, orig_time = None, None

                measures = [ffsa, mmifs, nmifs, lcfs, mrmr, mrmr2, orig]
                for i, result in enumerate(Parallel(return_as="generator")(delayed(cl.transform)() for cl in measures)):
                    match i:
                        case 0:
                            S_ffsa, _, ffsa_time = result
                            if self.verbose > 0: print(f"{datetime.now()}: FFSA done")
                        case 1:
                            S_mmifs, _, mmifs_time = result
                            if self.verbose > 0: print(f"{datetime.now()}: MMIFS done")
                        case 2:
                            S_nmifs, _, nmifs_time = result
                            if self.verbose > 0: print(f"{datetime.now()}: NMIFS done")
                        case 3:
                            S_lcfs, _, lcfs_time = result
                            if self.verbose > 0: print(f"{datetime.now()}: LCFS done")
                        case 4:
                            S_mrmr, _, mrmr_time = result
                            if self.verbose > 0: print(f"{datetime.now()}: mRMR done")
                        case 5:
                            S_mrmr2, _, mrmr2_time = result
                            if self.verbose > 0: print(f"{datetime.now()}: mRMR# done")
                        case 6:
                            S_orig, orig_time = result
                            if self.verbose > 0: print(f"{datetime.now()}: ORIG done")

                measures_time = time.time() - measures_start
                self.measures_times.append(measures_time)
                if self.verbose > 0: print(f"{datetime.now()}: All done in: {measures_time}")

                if self.iter_selected is None:
                    self.iter_selected = {
                        fcbf._name: [],
                        fcbf2._name: [],
                        fcbfip._name: [],
                        ffsa._name: [],
                        mmifs._name: [],
                        nmifs._name: [],
                        lcfs._name: [],
                        mrmr._name: [],
                        f"{mrmr2._name}#": [],
                        orig._name: []
                    }

                self.iter_selected[ffsa._name].append(list(S_ffsa - self.selected))
                self.iter_selected[mmifs._name].append(list(S_mmifs - self.selected))
                self.iter_selected[nmifs._name].append(list(S_nmifs - self.selected))
                self.iter_selected[lcfs._name].append(list(S_lcfs - self.selected))
                self.iter_selected[mrmr._name].append(list(S_mrmr - self.selected))
                self.iter_selected[f"{mrmr2._name}#"].append(list(S_mrmr2 - self.selected))
                self.iter_selected[orig._name].append(list(S_orig - self.selected))

                ### FCBF ###
                if len(S_ordered_fcbf) > 0:
                    if S_ordered_fcbf[0] not in self.selected:
                        self.iter_selected[fcbf._name].append([S_ordered_fcbf[0]])
                    else:
                        S_ordered_fcbf = np.delete(S_ordered_fcbf, np.isin(S_ordered_fcbf, list(self.selected)))
                        if len(S_ordered_fcbf) > 0:
                            self.iter_selected[fcbf._name].append([S_ordered_fcbf[0]])
                        else:
                            self.iter_selected[fcbf._name].append([])
                else:
                    self.iter_selected[fcbf._name].append([])

                ### FCBF# ###
                if len(S_ordered_fcbf2) > 0:
                    if S_ordered_fcbf2[0] not in self.selected:
                        self.iter_selected[fcbf2._name].append([S_ordered_fcbf2[0]])
                    else:
                        S_ordered_fcbf2 = np.delete(S_ordered_fcbf2, np.isin(S_ordered_fcbf2, list(self.selected)))
                        if len(S_ordered_fcbf2) > 0:
                            self.iter_selected[fcbf2._name].append([S_ordered_fcbf2[0]])
                        else:
                            self.iter_selected[fcbf2._name].append([])
                else:
                    self.iter_selected[fcbf2._name].append([])

                ### FCBFiP ###
                if len(S_ordered_fcbfip) > 0:
                    if S_ordered_fcbfip[0] not in self.selected:
                        self.iter_selected[fcbfip._name].append([S_ordered_fcbfip[0]])
                    else:
                        S_ordered_fcbfip = np.delete(S_ordered_fcbfip, np.isin(S_ordered_fcbfip, list(self.selected)))
                        if len(S_ordered_fcbfip) > 0:
                            self.iter_selected[fcbfip._name].append([S_ordered_fcbfip[0]])
                        else:
                            self.iter_selected[fcbfip._name].append([])
                else:
                    self.iter_selected[fcbfip._name].append([])

                nn_time_start = time.time()
                selected, loss, accuracy = self.evaluate()
                nn_time_final = time.time() - nn_time_start

                if self.verbose > 0: print(f"{datetime.now()}: Training time: {nn_time_final}")
                self.nn_times.append(nn_time_final)
                if self.verbose > 0: print(f"{datetime.now()}: Selected feature in {len(self.selected)}. iteration:", selected)

                self.selected.add(selected)
                self.selected_sorted.append(selected)

                self.loss.append(loss)
                self.accuracy.append(accuracy)

        transform_time = time.time() - transform_time_start
        if not self.is_in_pipeline:
            if self.verbose > 0: print(f"{datetime.now()}: Transform time: {transform_time}")
            return self.selected_sorted, self.loss, self.accuracy, self.iter_selected, self.nn_times, transform_time
        else:
            if self.verbose > 0: print(f"{datetime.now()}: Transform time: {transform_time}")
            return X[list(self.selected)], y

    def nn_one_fold(self, candidate: int, train_index: np.ndarray, test_index: np.ndarray,
                    nn_layers: list[[int, Callable[[float], float]|None], ]) -> tuple[int, float, float]:
        """
        One fold of an evaluation. Used for parallel execution of the evaluation phase if CPU is used.

        :param candidate: Candidate feature index.
        :type candidate: int
        :param train_index: Row index of train samples.
        :type train_index: np.ndarray
        :param test_index: Row index of test samples.
        :type test_index: np.ndarray
        :param nn_layers: Layers of the neural network.
        :type nn_layers: list[[int, Callable[[float], float]|None], ]
        :return: Candidate index, loss and accuracy associated with the candidate.
        :rtype: tuple[int, float, float]
        """

        X_s = self.X_n[:, list(self.selected) + [candidate]]

        model = AgiLM(layers=nn_layers, tau=1, weights_boundary=[-0.1, 0.1])

        random.seed(time.time_ns())
        random.shuffle(train_index)
        random.shuffle(test_index)

        X_train, X_test = X_s[train_index], X_s[test_index]
        y_train, y_test = self.y_n[train_index].reshape(-1, 1), self.y_n[test_index].reshape(-1, 1)

        X_val = X_test
        y_val = y_test

        model.fit(train_inputs=X_train,
                  train_targets=y_train,
                  val_inputs=X_val,
                  val_targets=y_val,
                  mu=1e-3,
                  initial_tau=1,
                  max_iteration=1000,
                  early_max_stepsize=6,
                  fix=True,
                  MSE_training=True)

        metrics = model.score(X_test, y_test)
        metrics.classes = np.unique(self.y_n)

        return candidate, metrics.scaled_root_mean_squared(), metrics.accuracy_reg()

    def evaluate(self) -> tuple[int, float | np.floating, float | np.floating]:
        """
        Evaluates the selected feature set using a specific evaluator.

        :return: The newly selected feature, loss, accuracy.
        :rtype: tuple[int, float, float]
        """

        X = self.X_n
        y = self.y_n

        classes = np.unique(y)

        cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)

        nn_layers = [[len(self.selected) + 1, None], [8, sigmoid], [1, sigmoid]]
        sum = 0
        for i in range(1, len(nn_layers) - 1):
            sum += nn_layers[i][0] * nn_layers[i + 1][0]
        jacobi_size = X.shape[0] * (len(self.selected) + 1 + sum)

        candidate_loss: {int: float} = {}
        candidate_accuracy: {int: float} = {}

        if jacobi_size >= CONST_GPU_THRESHOLD:
            for measure in self.iter_selected.keys():
                candidate_list = self.iter_selected[measure][-1]

                if len(candidate_list) > 0: candidate = candidate_list[0]
                else: continue

                if candidate in candidate_loss: continue

                fold_loss = []
                fold_accuracy = []
                X_s = X[:, list(self.selected) + [candidate]]
                for i, (train_index, test_index) in enumerate(cv.split(X_s, self.y_d)):
                    model = AgiLM(layers = nn_layers, tau = 1, weights_boundary = [-0.1, 0.1])

                    if self.verbose > 0: print(f"{datetime.now()}: Fold {i + 1}")

                    random.seed(time.time_ns())
                    random.shuffle(train_index)
                    random.shuffle(test_index)

                    X_train, X_test = X_s[train_index], X_s[test_index]
                    y_train, y_test = y[train_index].reshape(-1, 1), y[test_index].reshape(-1, 1)

                    X_val = X_test
                    y_val = y_test

                    model.fit(train_inputs = X_train,
                              train_targets = y_train,
                              val_inputs = X_val,
                              val_targets = y_val,
                              mu = 1e-3,
                              initial_tau = 1,
                              max_iteration = 1000,
                              early_max_stepsize = 6,
                              fix = True,
                              MSE_training = True)

                    metrics = model.score(X_test, y_test)
                    metrics.classes = classes

                    fold_loss.append(metrics.scaled_root_mean_squared())
                    fold_accuracy.append(metrics.accuracy_reg())
                    if self.verbose > 1: print(f"{datetime.now()}: Feature {candidate} fold {i + 1} loss: {fold_loss[-1]}")
                    if self.verbose > 1: print(f"{datetime.now()}: Feature {candidate} fold {i + 1} accuracy: {fold_accuracy[-1]}")

                candidate_loss[candidate] = np.mean(fold_loss)
                candidate_accuracy[candidate] = np.mean(fold_accuracy)
                if self.verbose > 0: print(f"{datetime.now()}: Scaled RMSE for feature {candidate}:", candidate_loss[candidate])
                if self.verbose > 0: print(f"{datetime.now()}: Accuracy for feature {candidate}:", candidate_accuracy[candidate])

        else:
            cv_indices = dict({})
            for measure in self.iter_selected.keys():
                candidate_list = self.iter_selected[measure][-1]

                if len(candidate_list) > 0: candidate = candidate_list[0]
                else: continue

                if candidate in cv_indices: continue
                else: cv_indices[candidate] = list(cv.split(X, self.y_d))

            candidate_all_loss: {int: list[float]} = {k: [] for k in cv_indices.keys()}
            candidate_all_accuracy: {int: list[float]} = {k: [] for k in cv_indices.keys()}

            for result in Parallel(return_as="generator")(delayed(self.nn_one_fold)(c, f[0], f[1], nn_layers) for c in cv_indices.keys() for f in cv_indices[c]):
                candidate_all_loss[result[0]].append(result[1])
                candidate_all_accuracy[result[0]].append(result[2])

            for k in candidate_all_loss.keys():
                candidate_loss[k] = np.mean(candidate_all_loss[k])
                if self.verbose > 0: print(f"{datetime.now()}: Scaled RMSE for feature {k}: {candidate_loss[k]}")

                candidate_accuracy[k] = np.mean(candidate_all_accuracy[k])
                if self.verbose > 0: print(f"{datetime.now()}: Accuracy for feature {k}: {candidate_accuracy[k]}")

        selected = min(candidate_loss, key = candidate_loss.get)
        if self.verbose > 0: print(f"{datetime.now()}: Selected feature: {selected}\n")

        return selected, candidate_loss[selected], candidate_accuracy[selected]