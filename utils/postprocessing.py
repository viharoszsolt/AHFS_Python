import os
import time
import warnings

import numpy as np
from numpy import ndarray
import pandas as pd
from pymatreader import read_mat

import matplotlib.pyplot as plt
import seaborn as sns

class aggregated_metrics:
    def __init__(self, feature_dominance: bool = True, acc_err: bool = True, time: bool = True):
        """
        Gathers feature order and priority, accuracy, and error over multiple AHFS runs from logs, saving results in figures.

        :param feature_dominance: If True, calculates feature dominance and related score. Default value is True.
        :type feature_dominance: bool
        :param acc_err: If True, calculates accuracy and error mean. Default value is True.
        :type acc_err: bool
        :param time: If True, aggregates evaluation and total runtime. Default value is True.
        :type time: bool
        """

        self.feature_dominance = feature_dominance
        self.acc_err = acc_err
        self.time = time

        if not (self.feature_dominance or self.acc_err): raise ValueError("At least one parameter must be true!")

    def read_txt_logs(self, superdir: str):
        """
        Format of directory naming should follow convention set in logs.py.

        :param superdir: Path of directory which contains subdirectories, each being a different AHFS run.
        :type superdir: str

        :return: If feature_dominance and acc_err is True: a tuple containing dictionaries of all per-iteration selected features, per-iteration accuracy, per-iteration error, and feature count for each dataset.
                 If only acc_err is True: a tuple containing dictionaries of per-iteration accuracy, per-iteration error, and feature count for each dataset.
                 If only feature_dominance is True: a tuple containing dictionaries of all per-iteration selected features and feature count for each dataset.
        """

        iter_dirs: list[str] = [f for f in os.listdir(superdir) if os.path.isdir(os.path.join(superdir, f))]
        #datasets: list[str] = [iter_dirs[i].split('_')[1].split('-')[0] for i in range(len(iter_dirs))]
        datasets: list[str] = [iter_dirs[i].split('-')[0] for i in range(len(iter_dirs))]
        feat_count: dict[str, int] = {k: -1 for k in np.unique(datasets)}

        if self.feature_dominance:
            all_selected: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}

        if self.acc_err:
            all_acc: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}
            all_err: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}

        if self.time:
            all_evaltime: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}
            all_runtime: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}

        for i, folder in enumerate(iter_dirs):
            if self.feature_dominance:
                with open(f"{superdir}/{folder}/features.txt", "rt") as f:
                    selected = f.readline().split("\t")
                    if selected[-1] == '\n': selected = selected[:-1]

                    if feat_count[datasets[i]] == -1:
                        feat_count[datasets[i]] = len(selected)
                    elif feat_count[datasets[i]] != len(selected):
                        raise ValueError(
                            f"Feature count difference at {superdir}/{folder}; expected {feat_count[datasets[i]]}, got {len(selected)}")

                all_selected[datasets[i]] = np.append(all_selected[datasets[i]],
                                                      np.array([int(selected[i]) for i in range(len(selected))]))

            if self.acc_err:
                for file in ["accuracy", "loss"]:
                    with open(f"{superdir}/{folder}/{file}.txt", "rt") as f:
                        selected = f.readline().split("\t")
                        if selected[-1] == '\n': selected = selected[:-1]

                        if feat_count[datasets[i]] == -1:
                            feat_count[datasets[i]] = len(selected)
                        elif feat_count[datasets[i]] != len(selected):
                            raise ValueError(
                                f"{file}.txt count difference at {superdir}/{folder}; expected {feat_count[datasets[i]]}, got {len(selected)}")

                    if file == "loss":
                        all_err[datasets[i]] = np.append(all_err[datasets[i]],
                                                         np.array([float(selected[i]) for i in range(len(selected))]))
                    else:
                        all_acc[datasets[i]] = np.append(all_acc[datasets[i]],
                                                         np.array([float(selected[i]) for i in range(len(selected))]))

            if self.time:
                evaltimes = pd.read_csv(f"{superdir}\\{folder}\\times.txt", skiprows=[0, 2, 3], delimiter='\t', header=None).dropna(axis=1).values.reshape(-1,)
                all_evaltime[datasets[i]] = np.append(all_evaltime[datasets[i]], evaltimes)

                full_time = pd.read_csv(f"{superdir}\\{folder}\\times.txt", skiprows=[0, 1, 2], delimiter='\t', header=None).dropna(axis=1).values.reshape(-1,)[0]
                all_runtime[datasets[i]] = np.append(all_runtime[datasets[i]], full_time)

        if self.feature_dominance and self.acc_err and self.time:
            return all_selected, all_acc, all_err, feat_count, all_evaltime, all_runtime
        if self.feature_dominance and self.acc_err:
            return all_selected, all_acc, all_err, feat_count
        elif self.acc_err:
            return all_acc, all_err, feat_count
        else:
            return all_selected, feat_count

    def read_mat_logs(self, superdir: str):
        """
        File naming format is {dataset_name}-{log_type}_{arbitrary_text}.mat. Recognized log types are: E, fo, RR. RR should have no separator and arbitrary text before it and the file extension.

        :param superdir: A directory containing all .mat log files.
        :type superdir: str

        :return: A tuple of dictionaries of all per-iteration selected features, per-iteration accuracy, per-iteration error, and feature count for each dataset.
        :rtype: tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]
        """

        items = [f for f in os.listdir(superdir) if f[-4:] == ".mat"]
        datasets = [i.split("-")[0] for i in items]

        all_selected: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}
        all_acc: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}
        all_err: dict[str, np.ndarray] = {k: np.array([]) for k in np.unique(datasets)}
        feat_count: dict[str, int] = {k: -1 for k in np.unique(datasets)}

        for idx, i in enumerate(items):
            mat = read_mat(f"{superdir}\\{i}")

            f_count = -1
            if i.split("-")[1] == "RR.mat":
                try:
                    f_count = mat["RR_final"].shape[1]
                except IndexError:
                    f_count = mat["RR_final"].shape[0]

                all_acc[datasets[idx]] = np.append(all_acc[datasets[idx]], mat["RR_final"])

            elif i.split("-")[1].split("_")[0] == "E":
                f_count = mat["E"].shape[0]
                all_err[datasets[idx]] = np.append(all_err[datasets[idx]], mat["E"])

            elif i.split("-")[1].split("_")[0] == "fo":
                f_count = mat["fo"].shape[0]
                all_selected[datasets[idx]] = np.append(all_selected[datasets[idx]], mat["fo"])

            if f_count != -1:
                if feat_count[datasets[idx]] == -1:
                    feat_count[datasets[idx]] = f_count
                elif feat_count[datasets[idx]] != f_count:
                    raise ValueError(f"Feature count does not match previous counts for {datasets[idx]} in file {i}!")

        return all_selected, all_acc, all_err, feat_count

    def run(self, feat_count: dict[str, int], save_path: str, all_selected: np.ndarray | None = None, all_acc: np.ndarray | None = None, all_err: np.ndarray | None = None) -> None:
        """
        Plots all results obtained and saves figures at the specified path.

        :param feat_count: Number of selected features for all datasets.
        :type feat_count: dict[str, int]
        :param save_path: Directory where all figures should be saved.
        :type save_path: str
        :param all_selected: Dictionary of per-iteration selected features for each dataset. If feature_dominance is True, an error is thrown if this variable is None. Default value is None.
        :type all_selected: np.ndarray | None
        :param all_acc: Dictionary of per-iteration accuracy for each dataset. If acc_err is True, an error is thrown if this variable is None. Default value is None.
        :type all_acc: np.ndarray | None
        :param all_err: Dictionary of per-iteration loss for each dataset. If acc_err is True, an error is thrown if this variable is None. Default value is None.
        :type all_err: np.ndarray | None

        :return: None
        """

        for k in all_selected.keys():
            if self.feature_dominance:
                all_selected[k] = all_selected[k].reshape(-1, feat_count[k]).astype(int)
                feature_points: np.ndarray = np.zeros(shape=(feat_count[k],), dtype=int)

                for i in range(len(feature_points)):
                    feature_points[i] = np.sum(len(feature_points) - np.argwhere(all_selected[k] == (i + 1))[:, 1])

                skip_flag = False
                count = np.zeros(all_selected[k].shape[1], dtype=int)
                for i in range(all_selected[k].shape[0]):
                    occ = np.bincount(all_selected[k][i, :] - 1)

                    try:
                        count += occ
                    except ValueError:
                        warnings.warn(f"Dataset {k} has faulty selected feature count, skipping! This occurs when the number of selected features does not equal the dataset's number of features.")

                        skip_flag = True
                        break

                if skip_flag: continue

                feat_x = np.arange(len(feature_points))

                fig, ax1 = plt.subplots()
                fig.set_size_inches(w=6.4 * (1.4 + feat_count[k] / 100), h=4.8)
                plt.xticks(feat_x, labels=[str(i + 1) for i in np.argsort(feature_points)[::-1]])

                ax1.plot(feat_x, feature_points[np.argsort(feature_points)[::-1]], color="red")
                ax1.set_ylabel("Feature score", color="red")
                ax1.tick_params(axis='y', labelcolor="red")
                ax1.set_xlabel("Feature")

                ax2 = ax1.twinx()
                ax2.plot(feat_x, count[np.argsort(feature_points)[::-1]])
                ax2.set_ylabel("Frequency of selection", color="blue")
                ax2.tick_params(axis='y', labelcolor="blue")
                ax2.set_ylim(top=np.max(count) * 1.01)

                plt.title(f"Feature scores and frequency for dataset {k}")

                plt.savefig(f"{save_path}/{k}-feature_scores-{int(time.time())}.png")

            if self.acc_err:
                all_acc[k] = all_acc[k].reshape(-1, feat_count[k]).astype(float)
                all_err[k] = all_err[k].reshape(-1, feat_count[k]).astype(float)

                err_mean: np.ndarray = np.mean(all_err[k], axis=0)
                acc_mean: np.ndarray = np.mean(all_acc[k], axis=0)

                x = np.arange(err_mean.shape[0])

                fig, ax1 = plt.subplots()
                fig.set_size_inches(w=6.4 * (1.4 + feat_count[k] / 100), h=4.8)

                ax1.plot(x + 1, err_mean, "bo-")
                ax1.set_ylabel("RMSE", color="blue")
                ax1.tick_params(axis='y', labelcolor="blue")
                ax1.set_xlabel("Iteration")

                ax2 = ax1.twinx()
                ax2.plot(x + 1, acc_mean, "ro-")
                ax2.set_ylabel("Accuracy", color="red")
                ax2.tick_params(axis='y', labelcolor="red")

                plt.title(f"Loss and accuracy for dataset {k}")

                plt.savefig(f"{save_path}/{k}-err_acc-{int(time.time())}.png")

    def compare_two(self, set_a: dict[str, dict | str], set_b: dict[str, dict | str], feat_count: dict[str, int], save_path: str) -> None:
        """
        Compares two dictionaries of AHFS evaluations, plots the differences and saves them as figures. Only feature scoring and loss values are compared.

        :param set_a: Dictionary containing all_selected, all_err, and label keys, latter being a string. Former two values should follow the format of variables returned by read_mat_logs or read_txt_logs.
        :type set_a: dict[str, dict | str]
        :param set_b: See set_a. Ensure identical keys within the set dictionaries for dataset naming.
        :type set_b: dict[str, dict | str]
        :param feat_count: Number of selected features for all datasets.
        :type feat_count: dict[str, int]
        :param save_path: Directory where all figures should be saved.
        :type save_path: str

        :return: None
        """

        for k in feat_count.keys():
            if self.feature_dominance:
                set_a["all_selected"][k] = set_a["all_selected"][k].reshape(-1, feat_count[k]).astype(int)
                set_b["all_selected"][k] = set_b["all_selected"][k].reshape(-1, feat_count[k]).astype(int)
                feature_points_a: np.ndarray = np.zeros(shape=(feat_count[k],), dtype=int)
                feature_points_b: np.ndarray = np.zeros(shape=(feat_count[k],), dtype=int)

                for i in range(len(feature_points_a)):
                    feature_points_a[i] = np.sum(len(feature_points_a) - np.argwhere(set_a["all_selected"][k] == (i + 1))[:, 1])
                    feature_points_b[i] = np.sum(len(feature_points_b) - np.argwhere(set_b["all_selected"][k] == (i + 1))[:, 1])

                feat_x = np.arange(len(feature_points_a))

                plt_data = pd.DataFrame({
                    "Feature": np.tile(feat_x+1, 2),
                    "Feature score": np.concatenate([feature_points_a[np.argsort(feature_points_a)[::-1]], feature_points_b[np.argsort(feature_points_a)[::-1]]]),
                    "Set": [set_a["label"]] * len(feature_points_a) + [set_b["label"]] * len(feature_points_b)
                })

                plt.figure(figsize=(6.4 * (1.4 + feat_count[k] / 100), 4.8))
                ax1 = sns.lineplot(data=plt_data, x="Feature", y="Feature score", hue="Set", marker="o", linewidth=3)
                ax1.set_xticks(feat_x+1)
                ax1.set_xticklabels((feat_x+1)[np.argsort(feature_points_a)[::-1]])
                ax1.tick_params(axis='both', which='major', labelsize=18)
                ax1.set_xlabel("Feature", fontsize=20)
                ax1.set_ylabel("Feature score", fontsize=20)
                plt.setp(ax1.get_legend().get_texts(), fontsize='20')
                plt.setp(ax1.get_legend().get_title(), fontsize='20')
                plt.tight_layout()

                plt.savefig(f"{save_path}/{k}-feature_scores_sets-{int(time.time())}.png")

            if self.acc_err:
                all_err_a = set_a["all_err"][k].reshape(-1, feat_count[k]).astype(float)
                all_err_b = set_b["all_err"][k].reshape(-1, feat_count[k]).astype(float)

                x = np.arange(all_err_a.shape[1])

                plt_data = pd.DataFrame({
                    "Iteration": np.tile(x + 1, all_err_a.shape[0] + all_err_b.shape[0]),
                    "SRMSE": np.concatenate([all_err_a.flatten(), all_err_b.flatten()]),
                    "Set": [set_a["label"]] * all_err_a.size + [set_b["label"]] * all_err_b.size
                })

                plt.figure(figsize=(6.4 * (1.4 + feat_count[k] / 100), 4.8))
                ax1 = sns.lineplot(data=plt_data, x="Iteration", y="SRMSE", hue="Set", marker="o", errorbar="sd", linewidth=3)
                ax1.tick_params(axis='both', which='major', labelsize=18)
                ax1.set_xlabel("Iteration", fontsize=20)
                ax1.set_ylabel("SRMSE", fontsize=20)
                plt.setp(ax1.get_legend().get_texts(), fontsize='20')
                plt.setp(ax1.get_legend().get_title(), fontsize='20')
                plt.tight_layout()

                plt.savefig(f"{save_path}/{k}-err_sets-{int(time.time())}.png")
