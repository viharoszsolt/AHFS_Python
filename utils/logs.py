from os import makedirs
from time import time

def log_results(path: str, features: list[int], loss: list[float], accuracy: list[float], periter: dict[str: list[int]], eval_times: list[float], transform_time: float) -> None:
    """
    Logs AHFS results from a single run.

    :param path: Folder path to establish logs in, must not exist. When running AHFS multiple times, establish a superdirectory, e.g. "novelAHFS_runs/{dataset}". Epoch time is automatically concatenated to the end of the path to avoid overwrite. Folder naming should follow the format {arbitrary text}_{dataset name}_{arbitrary text} to ensure compatibility with other utility functions.
    :type path: str
    :param features: Ordered list of selected features.
    :type features: list[int]
    :param loss: Ordered list of loss per iteration.
    :type loss: list[float]
    :param accuracy: Ordered list of accuracy per iteration.
    :type accuracy: list[float]
    :param periter: Dictionary with measure names as keys and ordered list of candidate features suggested by the measure.
    :type periter: dict[str: list[int]]
    :param eval_times: Ordered list of evaluation time per iteration.
    :type eval_times: list[float]
    :param transform_time: Transform time.
    :type transform_time: float
    :return: None
    """

    full_path = f"{path}-{int(time())}"
    makedirs(full_path)

    with open(f"{full_path}/periter_features.csv", "w", newline="\n") as f:
        for k in periter.keys():
            f.write(f"{k},")
        f.write("\n")

        for i in range(len(features)):
            for k in periter.keys():
                try:
                    f.write(f"{periter[k][i][0] + 1},")
                except IndexError:
                    f.write("-,")
            f.write("\n")

    with open(f"{full_path}/features.txt", "a") as f:
        for i in features:
            f.write(f"{i + 1}\t")
        f.write("\n")

    with open(f"{full_path}/loss.txt", "a") as f:
        for i in loss:
            f.write(f"{i}\t")
        f.write("\n")

    with open(f"{full_path}/accuracy.txt", "a") as f:
        for i in accuracy:
            f.write(f"{i}\t")
        f.write("\n")

    with open(f"{full_path}/times.txt", "a") as f:
        f.write("eval times:\n")
        for i in eval_times:
            f.write(f"{i}\t")
        f.write("\n")

        f.write("transform time:\n")
        f.write(f"{transform_time}\n")
