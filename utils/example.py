def run_one_dataset(path: str, target: str, size: int) -> None:
    import pandas as pd
    from ahfs_class.ahfs import AHFS

    """
    Runs one dataset. See utils/presets.py and utils/helpers/loader.py for more specific usage.

    :param path: Path to the .csv file.
    :type path: str
    :param target: Path to the target variable (assumed .csv).
    :type target: str
    :param size: Number of features to select.
    :type size: int
    
    :return: None
    """

    data = pd.read_csv(path).values
    target = pd.read_csv(target).values

    ahfs = AHFS(size)
    sel, loss, acc, perit = ahfs.transform(data, target)

def run_all_presets() -> None:
    """
    Runs all preset datasets with their fixed configuration 5 times each, logging each run.

    :return: None
    """

    import time
    import utils.presets as p
    from utils.logs import log_results

    datasets = [
        p.CarDataset, p.WisconsinBreastCancerDataset, p.BankDataset, p.ForestFiresDataset,
        p.AbaloneDataset, p.CommCrimeDataset, p.SonarDataset, p.YearPredDataset,
        p.ReplicatedParkinsonDataset, p.SuperConductDataset, p.CalculatedCuttingFcDataset,
        p.CalculatedCuttingPDataset, p.CalculatedCuttingTDataset, p.CalculatedCuttingRaDataset,
        p.HousingDataset, p.IrisDataset, p.MeasuredCuttingFcDataset, p.MeasuredCuttingPDataset,
        p.MeasuredCuttingTDataset, p.MeasuredCuttingRaDataset, p.ParkinsonsTelemonitoringMotorDataset,
        p.ParkinsonsTelemonitoringTotalDataset, p.WineDataset, p.WineQualityRedDataset,
        p.MitbihTest
    ]

    for ds in datasets:
        for _ in range(5):
            time.sleep(1)
            pst = ds()
            fo, err, acc, iter_feat, eval_time, transf_time = pst.run()

            log_results(f"novelahfs_run-{int(time.time())}\\{pst.name}", fo, err, acc, iter_feat, eval_time, transf_time)

def run_fo_eval() -> None:
    """
    Evaluate a feature order using the AgiLM evaluator.

    :return: None
    """

    import numpy as np
    import pandas as pd
    from utils.nn_test import AgiLM_test

    fo_iris = np.array([2, 3, 0, 1])
    path = "datasets\\iris_prep.csv"

    loss, acc = AgiLM_test(fo_iris, path, 5, 0)

    pd.DataFrame(loss).to_csv("iris_python_loss.csv", index=False)
    pd.DataFrame(acc).to_csv("iris_python_acc.csv", index=False)

def run_postprocessing_metrics() -> None:
    """
    Compare two feature selection results.

    :return: None
    """

    import numpy as np
    from utils.postprocessing import aggregated_metrics

    metrics = aggregated_metrics()

    path1 = "INSERT PATH1"
    path2 = "INSERT PATH2"

    set1 = dict({})
    set2 = dict({})

    set1["all_selected"], set1["all_acc"], set1["all_err"], count1, _, _ = metrics.read_txt_logs(path1)
    set2["all_selected"], set2["all_acc"], set2["all_err"], count2, _, _ = metrics.read_txt_logs(path2)

    s1_keys = np.array(list(count1.keys()))
    s2_keys = np.array(list(count2.keys()))

    for k in set2.keys():
        for i in range(len(s1_keys)):
            set2[k][s1_keys[i]] = set2[k].pop(s2_keys[i])

    for i in range(len(s1_keys)):
        count2[s1_keys[i]] = count2.pop(s2_keys[i])

    set1["label"] = "Set 1"
    set2["label"] = "Set 2"

    metrics.compare_two(set1, set2, count1, "INSERT SAVE PATH")