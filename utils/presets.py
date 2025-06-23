import numpy as np

from ahfs_class.ahfs import AHFS
from utils.helpers.loader import LoaderBase

class CarDataset(LoaderBase):
    def __init__(self):
        super().__init__("Car", "datasets/car_prep.csv", [22, 23, 24, 25], 0, drop_columns = [21])
        self.data = self.data.astype(np.int32)
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 21,
                data_bin = 0,
                target_bin = 0,
                save_precomp_path = "datasets/cars_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class WisconsinBreastCancerDataset(LoaderBase):
    def __init__(self):
        super().__init__("WisconsinBreastCancer", "datasets/wdbc_prep.csv", [31], 0, drop_columns = [30])
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 30,
                data_bin = 400,
                target_bin= 0,
                save_precomp_path = "datasets/wisconsin_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class BankDataset(LoaderBase):
    def __init__(self):
        super().__init__("Bank", "datasets/bank_prep.csv", [49], 0, drop_columns = [48])
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 48,
                data_bin = 0,
                target_bin = 0,
                save_precomp_path = "datasets/bank_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class ForestFiresDataset(LoaderBase):
    def __init__(self):
        super().__init__("ForestFires", "datasets/forestfires_prep.csv", [30], 0, drop_columns = [29])

        days = np.argmax(self.data[:, 0:7], axis = 1)
        months = np.argmax(self.data[:, 7:19], axis = 1)

        self.data = np.concatenate([days.reshape(-1, 1), months.reshape(-1, 1), self.data[:, 19:]], axis = 1)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 12,
                data_bin = 100,
                target_bin = 250,
                save_precomp_path = "datasets/forestfires_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class AbaloneDataset(LoaderBase):
    def __init__(self):
        super().__init__("Abalone", "datasets/abalone_prep.csv", [11], 0, drop_columns = [10])
        self.target = self.target.astype(np.int32)
        self.target[np.where(self.target == 29)] = 28
        self.target -= 1

        sex = np.argmax(self.data[:, 0:3], axis = 1)

        self.data = np.concatenate([sex.reshape(-1, 1), self.data[:, 3:]], axis = 1)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 8,
                data_bin = 500,
                target_bin = 0,
                save_precomp_path = "datasets/abalone_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class CommCrimeDataset(LoaderBase):
    def __init__(self):
        super().__init__("CommunitiesAndCrime", "datasets/communities_and_crime_prep.csv", [102], 0, drop_columns = [101])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 101,
                data_bin = 500,
                target_bin = 500,
                save_precomp_path = "datasets/communitiesandcrimes_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class SonarDataset(LoaderBase):
    def __init__(self):
        super().__init__("Sonar", "datasets/sonar_prep.csv", [61], 0, drop_columns = [60])
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 60,
                data_bin = 100,
                target_bin = 0,
                save_precomp_path = "datasets/sonar_measures.npy"
            )

        return ahfs.transform(self.data, self.target)
    
class YearPredDataset(LoaderBase):
    def __init__(self):
        super().__init__("YearPred", "datasets/year_prediction_prep.csv", [91], 0, drop_columns = [90])
        self.target = self.target.astype(np.int32)

        year_indexes = np.argwhere(np.unique(self.target).reshape(-1, 1) == self.target)
        self.target[year_indexes[:, 1]] = year_indexes[:, 0]

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 90,
                data_bin = 2500,
                target_bin = 0,
                save_precomp_path = "datasets/yearprediction_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class ReplicatedParkinsonDataset(LoaderBase):
    def __init__(self):
        super().__init__("ReplicatedParkinson", "datasets/replicated_parkinson_prep.csv", [46], 0, drop_columns = [45])
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 45,
                data_bin = 40,
                target_bin = 0,
                save_precomp_path = "datasets/replicatedparkinson_measures.npy"
            )

        return ahfs.transform(self.data, self.target)
    
class SuperConductDataset(LoaderBase):
    def __init__(self):
        super().__init__("SuperConductivity", "datasets/superconductivity_prep.csv", [82], 0, drop_columns = [81])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 81,
                data_bin = 1200,
                target_bin = 800,
                save_precomp_path = "datasets/superconductivity_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class CalculatedCuttingFcDataset(LoaderBase):
    def __init__(self):
        super().__init__("CalculatedCuttingFc", "datasets/calculated_cutting_prep.csv", [6], 0, drop_columns = [5, 7, 8, 9])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 5,
                data_bin = 180,
                target_bin = 4,
                save_precomp_path = "datasets/calculatedcuttingfc_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class CalculatedCuttingPDataset(LoaderBase):
    def __init__(self):
        super().__init__("CalculatedCuttingP", "datasets/calculated_cutting_prep.csv", [7], 0, drop_columns = [5, 6, 8, 9])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 5,
                data_bin = 180,
                target_bin = 4,
                save_precomp_path = "datasets/calculatedcuttingp_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class CalculatedCuttingTDataset(LoaderBase):
    def __init__(self):
        super().__init__("CalculatedCuttingT", "datasets/calculated_cutting_prep.csv", [8], 0, drop_columns = [5, 6, 7, 9])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 5,
                data_bin = 180,
                target_bin = 4,
                save_precomp_path = "datasets/calculatedcuttingt_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class CalculatedCuttingRaDataset(LoaderBase):
    def __init__(self):
        super().__init__("CalculatedCuttingRa", "datasets/calculated_cutting_prep.csv", [9], 0, drop_columns = [5, 6, 7, 8])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 5,
                data_bin = 180,
                target_bin = 2,
                save_precomp_path = "datasets/calculatedcuttingra_measures.npy"
            )

        return ahfs.transform(self.data, self.target)
    
class HousingDataset(LoaderBase):
    def __init__(self):
        super().__init__("Housing", "datasets/housing_prep.csv", [14], 0, drop_columns = [13])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 13,
                data_bin = 50,
                target_bin = 100,
                save_precomp_path = "datasets/housing_measures.npy"
            )

        return ahfs.transform(self.data, self.target)
    
class IrisDataset(LoaderBase):
    def __init__(self):
        super().__init__("Iris", "datasets/iris_prep.csv", [5, 6, 7], 0, drop_columns = [4])
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 4,
                data_bin = 15,
                target_bin = 0,
                save_precomp_path = "datasets/iris_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class MeasuredCuttingFcDataset(LoaderBase):
    def __init__(self):
        super().__init__("MeasuredCuttingFc", "datasets/measured_cutting_prep.csv", [4], 0, drop_columns = [3, 5, 6, 7])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 3,
                data_bin = 5,
                target_bin = 3,
                save_precomp_path = "datasets/measuredcuttingfc_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class MeasuredCuttingPDataset(LoaderBase):
    def __init__(self):
        super().__init__("MeasuredCuttingP", "datasets/measured_cutting_prep.csv", [5], 0, drop_columns = [3, 4, 6, 7])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 3,
                data_bin = 5,
                target_bin = 3,
                save_precomp_path = "datasets/measuredcuttingp_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class MeasuredCuttingTDataset(LoaderBase):
    def __init__(self):
        super().__init__("MeasuredCuttingT", "datasets/measured_cutting_prep.csv", [6], 0, drop_columns = [3, 4, 5, 7])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 3,
                data_bin = 5,
                target_bin = 2,
                save_precomp_path = "datasets/measuredcuttingt_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class MeasuredCuttingRaDataset(LoaderBase):
    def __init__(self):
        super().__init__("MeasuredCuttingRa", "datasets/measured_cutting_prep.csv", [7], 0, drop_columns = [3, 4, 5, 6])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 3,
                data_bin = 5,
                target_bin = 3,
                save_precomp_path = "datasets/measuredcuttingra_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class ParkinsonsTelemonitoringMotorDataset(LoaderBase):
    def __init__(self):
        super().__init__("ParkinsonsTelemonitoringMotor", "datasets/parkinsons_telemonitoring_prep.csv", [21], 0, drop_columns = [0, 20, 22])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 19,
                data_bin = 750,
                target_bin = 5,
                save_precomp_path = "datasets/parkinsonstelemonitoringmotor_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class ParkinsonsTelemonitoringTotalDataset(LoaderBase):
    def __init__(self):
        super().__init__("ParkinsonsTelemonitoringTotal", "datasets/parkinsons_telemonitoring_prep.csv", [22], 0, drop_columns = [0, 20, 21])

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 19,
                data_bin = 750,
                target_bin = 5,
                save_precomp_path = "datasets/parkinsonstelemonitoringtotal_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class WineDataset(LoaderBase):
    def __init__(self):
        super().__init__("Wine", "datasets/wine_prep.csv", [14], 0, drop_columns = [13])
        self.target = self.target.astype(np.int32)
        self.target -= 1

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 13,
                data_bin = 50,
                target_bin = 0,
                save_precomp_path = "datasets/wine_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class WineQualityRedDataset(LoaderBase):
    def __init__(self):
        super().__init__("WineQualityRed", "datasets/wine_quality_red_prep.csv", [12], 0, drop_columns = [11])
        self.target = self.target.astype(np.int32)
        self.target -= 3

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 11,
                data_bin = 50,
                target_bin = 0,
                save_precomp_path = "datasets/winequalityred_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class MitbihTest(LoaderBase):
    def __init__(self):
        super().__init__("MitbihTest", "datasets/mitbih_test.csv", [187], None)
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 187,
                data_bin = 20,
                target_bin = 0,
                save_precomp_path = "datasets/mitbihtest_measures.npy"
            )

        return ahfs.transform(self.data, self.target)

class MNISTTrain(LoaderBase):
    def __init__(self):
        super().__init__("MNISTTrain", "datasets/mnist_train.csv", [0], None)
        self.target = self.target.astype(np.int32)

    def run(self, **kwargs) -> tuple[list[int], list[float], list[float], dict[str: int]] | tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            ahfs = AHFS(**kwargs)
        else:
            ahfs = AHFS(
                k = 784,
                data_bin = 0,
                target_bin = 0,
                save_precomp_path = "datasets/mnisttrain_measures.npy"
            )

        return ahfs.transform(self.data, self.target)
