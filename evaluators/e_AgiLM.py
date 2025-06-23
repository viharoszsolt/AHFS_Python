from collections.abc import Callable
from evaluators.helpers.eval import EvalBase

import numpy as np

CONST_GPU_THRESHOLD = 230000

def sigmoid(x):
    sigm_x = 1 / (1 + np.exp(-x))
    return sigm_x

class AgiLM(EvalBase):
    def __init__(self, layers: list[[int, Callable[[float], float]|None], ], tau: float = 1.0,
                 weights_boundary: list[float] = [-0.1, 0.1]):
        """
        Implements the interface of an artificial neural network with Levenberg-Marquardt optimization made by VIHAROS, Zsolt János and SZŰCS, Ágnes.
        https://doi.org/10.48550/arXiv.2211.11491

        Further improvements to this implementation were done by HOANG, Anh Tuan; VINCZE, Tibor; GERCUJ, Bence.

        :param layers: A two-dimensional list describing the network architecture. Each sublist defines neuron count and activation function. Example: [[8, None], [4, sigmoid], [2, sigmoid]] describes an ANN with 8 inputs, one hidden layer with 4 neurons and sigmoid activation, and binary output with sigmoid activation. "sigmoid" is a Python function with signature (float) -> float.
        :type layers: list[[int, Callable[[float], float] | None], ]
        :param tau: Tau parameter controlling the optimizer. Default value is 1.0.
        :type tau: float
        :param weights_boundary: Boundary for weights. Default value is [-0.1, 0.1].
        :type weights_boundary: list[float, float]

        :return: None
        :rtype: None
        """

        super().__init__()
        self.layers = layers
        self.tau = tau
        self.weights_boundary = weights_boundary

        self.nn = None

    def fit(self, **kwargs) -> None:
        """
        Fits the neural network on the provided data.

        :param kwargs: See below for non-optional and relevant parameters. Consult the source code for the remaining arguments.

        :return: None
        :rtype: None

        :keyword train_inputs: Data for the network to be fitted on. np.ndarray
        :keyword train_targets: Target variable for training. np.ndarray
        :keyword val_inputs: Data to test per epoch. np.ndarray
        :keyword val_targets: Target variable for validation. np.ndarray
        :keyword mu: Optimizer parameter. float
        :keyword initial_tau: Optimizer parameter. float
        :keyword max_iteration: Maximum number of epochs for training. int
        :keyword early_max_stepsize: Early stopping patience. int
        :keyword MSE_training: Whether to use Mean-Squared Error as a loss function. bool
        :keyword fix: Whether to use the fixed version of the algorithm. bool

        """
        samples = kwargs["train_inputs"].shape[0]
        input_size = kwargs["train_inputs"].shape[1] * self.layers[0][0]

        sum = 0
        for i in range(1, len(self.layers) - 1):
            sum += self.layers[i][0] * self.layers[i+1][0]

        jacobi_size = samples * (input_size + sum)
        if jacobi_size >= CONST_GPU_THRESHOLD:
            from evaluators.src.LM.lm_torch import NeuralNetwork
        else:
            from evaluators.src.LM.lm import NeuralNetwork

        nn = NeuralNetwork(self.layers, self.tau, self.weights_boundary)
        nn.fit(**kwargs)

        self.nn = nn

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs prediction based on the input data. Fitting is needed prior to prediction.

        :param X: Data from which the target variable is predicted from.
        :type X: np.ndarray
        :return: Predicted target variable.
        :rtype: np.ndarray
        """
        if self.nn is None: raise ValueError("Neural network fit was not called!")

        return self.nn.evaluate(X)