import random
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from evaluators.e_AgiLM import AgiLM
from evaluators.e_AgiLM import sigmoid
from utils.preprocessing import discretize_X_y

def AgiLM_test(fo: np.ndarray, path: str, repeat: int, disc_y: int = 0) -> tuple[list[list[float]], list[list[float]]]:
    df = pd.read_csv(path)

    X = df.iloc[:, :-1].values
    y = df["target"].values

    if disc_y != 0:
        _, y = discretize_X_y(X, y, 0, disc_y)

    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    X_n = scaler.fit_transform(X)
    y_n = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1, )

    run_loss = []
    run_accuracy = []
    for r in range(repeat):
        print(f"Run {r + 1}")

        classes = np.unique(y_n)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        iter_loss = []
        iter_accuracy = []
        for j in range(fo.shape[0]):
            X_s = X_n[:, fo[:(j+1)]]

            fold_loss = []
            fold_accuracy = []
            for i, (train_index, test_index) in enumerate(cv.split(X_s, y)):
                model = AgiLM(layers=[[j + 1, None], [8, sigmoid], [1, sigmoid]],
                              tau=1, weights_boundary=[-0.1, 0.1])

                print(f"Fold {i + 1}")

                random.seed(time.time_ns())
                random.shuffle(train_index)
                random.shuffle(test_index)

                X_train, X_test = X_s[train_index], X_s[test_index]
                y_train, y_test = y_n[train_index].reshape(-1, 1), y_n[test_index].reshape(-1, 1)


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
                metrics.classes = classes

                fold_loss.append(metrics.scaled_root_mean_squared())
                fold_accuracy.append(metrics.accuracy_reg())

            iter_loss.append(np.mean(fold_loss))
            iter_accuracy.append(np.mean(fold_accuracy))

        run_loss.append(iter_loss)
        run_accuracy.append(iter_accuracy)

    return run_loss, run_accuracy