import pandas as pd
import numpy as np

from ahfs_class.ahfs import AHFS

class LoaderBase:
    def __init__(self, name: str, path: str, target: list[int], header: None|int|list[int] = None, sep: str = ",", drop_columns: None|list[int] = None, drop_rows: None|list[int] = None):
        """
        Base class for dataset processing and AHFS running. Only CSV files are supported.

        :param name: Name of the instance.
        :type name: str
        :param path: Path to the CSV dataset.
        :type path: str
        :param target: List of indexes that define target column(s). A list size greater than 1 implies one-hot encoding.
        :type target: list[int]
        :param header: List of indexes that define header row(s). If None, no header is extracted from the data. Default value is None.
        :type header: None | int | list[int]
        :param sep: Separator character. Default value is ",".
        :type sep: str
        :param drop_columns: List of indexes that define which column(s) to remove. If None, no removal is done. Default value is None.
        :type drop_columns: None | list[int]
        :param drop_rows: List of indexes that define which row(s) to remove. If None, no removal is done. Default value is None.
        :type drop_rows: None | list[int]
        """

        self.name: str = name
        self.data: np.ndarray = pd.read_csv(path, header=header, sep=sep, skiprows=drop_rows).values
        self.target: np.ndarray = np.squeeze(self.data[:, target])

        if drop_columns is not None:
            [drop_columns.append(i) for i in target]
            self.data = np.delete(self.data, drop_columns, 1)
        else:
            self.data = np.delete(self.data, target, 1)

        try:
            if self.target.shape[1] > 1: self.target = np.argmax(self.target, axis=1)
        except IndexError:
            pass

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        """
        Function for running an AHFS instance on the loaded dataset.

        :param kwargs: Overrides AHFS run parameters. See below for detailed documentation.
        :return: A list of the selected features, loss of the selected feature set, accuracy of the selected feature set, features selected per iteration.

        :keyword k: Number of features to select. int
        :keyword data_bin: How many bins to discretize the dataset into, excluding the target variable. If 0, no discretization is performed. Default value is 5. int
        :keyword target_bin: How many bins to discretize the target into. Effectively sets the number of classes. If 0, no discretization is performed. Default value is 2. int
        :keyword save_precomp: Whether to save all basic measures computed during the precomputing phase in a binary .npy file. Path is set by save_precomp_path. Default value is True. bool
        :keyword save_precomp_path: If save_precomp is True, sets the path for saving the precomputed basic measures. If this variable's value is None, saves into the current directory with filename format "measures_{time.time_ns()}.npy". Default value is None. str | None
        :keyword load_precomp_path: Path to load the precomputed basic measures from, skipping the precomputing phase. File must be a binary numpy file. If None, precomputing is not skipped. Default value is None. str | None
        :keyword is_in_pipeline: Set to True if the algorithm is intended to be part of a scikit-learn pipeline. Changes the return value of transform() to X and y. Default value is False. bool
        :keyword verbose: Controls global verbosity, including feature selection measures. 0: no output, 1: feature selection measure execution time, selected feature and metric, and basic algorithm steps are printed, 2: all steps are printed. Default value is 1. int
        """

        if len(kwargs) == 0: raise ValueError("Empty AHFS parameters dictionary!")

        ahfs = AHFS(**kwargs)
        sel, loss, acc, perit = ahfs.transform(self.data, self.target)

        return sel, loss, acc, perit

    def save(self, path: str|None = None) -> None:
        """
        Saves the transformed preset dataset into a .csv file.

        :param path: Path to save the dataset. If None, saved in the working directory. Default value is None.
        :type path: str | None
        """

        dataset = pd.DataFrame(self.data)
        dataset["target"] = self.target

        if path is None:
            dataset.to_csv(f"{self.name}_processed.csv", index=False)

        else:
            dataset.to_csv(f"{path}", index=False)