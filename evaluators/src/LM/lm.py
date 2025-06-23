import numpy as np
import copy

tr_Ek = []
vl_Eexpabs = []
vl_MSE = []
vl_RR = []
tr_MSE = []
tr_RR = []
tr_CE = []
vl_CE = []
Tau = []
Mu = []
Eta = []
gauss_grad = []

def sigmoid(x):
    sigm_x = 1 / (1 + np.exp(-x))
    return sigm_x

class OptimalValues:
    def __init__(self, layers, tau, initial_tau, fix, vl_Eexpabs, vl_MSE, vl_CE, vl_RR, mu, iter):
        self.layers = layers
        self.tau = tau
        self.fix = fix

        self.initial_tau = initial_tau

        self.vl_Eexpabs = vl_Eexpabs
        self.vl_MSE = vl_MSE
        self.vl_CE = vl_CE
        self.vl_RR = vl_RR

        self.mu = mu

        self.iter = iter

class Layer:
    def __init__(self, prev_node_number, node_number, activation_function, weights_boundary):
        self.activation = activation_function
        self.weights = np.random.uniform(weights_boundary[0], weights_boundary[1], (prev_node_number + 1, node_number)) # - 0.5
        self.weights_candidate = self.weights.copy()
        self.weights_optimal = self.weights.copy()

        self.sigmoid_ol = np.empty((node_number,)) #None
        self.prev_sigmoid_ol = np.empty((prev_node_number,))

class NeuralNetwork:

    def __init__(self, layers, tau, weights_boundary = [-0.5, 0.5]): #*args, **kwargs):
        self.weights_boundary = weights_boundary
        self.layers = self.create_layers(layers)

        self.tau = tau
        self.tau_candidate = tau

        self.delta_tau = 0

        self.optimal = None

        self.calculate_weights_number()

        self.train_Error = None
        self.validation_Error = None

    def create_layers(self, layers):
        return [Layer(layers[i-1][0], layers[i][0], layers[i][1], self.weights_boundary) for i in range(1, len(layers))]

    def forward_propagation(self, inputs, mode = 0):
        for ind, layer in enumerate(self.layers):
            inputs = np.hstack((np.ones((inputs.shape[0],1)), inputs))
            if mode == 0:
                layer.prev_sigmoid_ol = inputs.copy()
            if mode == 1:
                inputs = np.dot(inputs, layer.weights_candidate)
            elif mode == 0:
                inputs = np.dot(inputs, layer.weights)
            else:
                inputs = np.dot(inputs, self.optimal.layers[ind].weights)

            inputs = layer.activation(inputs)

            if mode == 0:
                layer.sigmoid_ol = inputs.copy()

        return inputs

    def d_ol_ol1_mx(self, layer):
        sigmoid_ol = layer.sigmoid_ol.copy()
        S_O = sigmoid_ol - np.square(sigmoid_ol)

        d_ol1 = S_O.reshape(S_O.shape[0], S_O.shape[1], 1) * layer.weights[1:].T.reshape(1, layer.weights[1:].shape[1],
                                                                                         layer.weights[1:].shape[0])
        return d_ol1

    def d_ol_wl_mx(self, layer):
        sigmoid_ol = layer.sigmoid_ol.copy()
        S_O = sigmoid_ol - np.square(sigmoid_ol)

        return S_O.reshape(S_O.shape[0], S_O.shape[1], 1) @ layer.prev_sigmoid_ol.reshape(
            layer.prev_sigmoid_ol.shape[0], 1, layer.prev_sigmoid_ol.shape[1])

    def calculate_weights_number(self):
        self.weights_number = 0
        for layer in self.layers:
            self.weights_number = self.weights_number + layer.weights.size

    def compute_Jacobian(self):
        d_ol_ol1 = np.ones((self.layers[-1].sigmoid_ol.shape[0], 1, 1, 1)) * np.identity(
            self.layers[-1].sigmoid_ol.shape[1]).reshape(1, self.layers[-1].sigmoid_ol.shape[1],
                                                         self.layers[-1].sigmoid_ol.shape[1], 1)

        Jacobian = np.empty((self.layers[-1].sigmoid_ol.shape[0], self.layers[-1].sigmoid_ol.shape[1], 0))

        for ind, layer in enumerate(reversed(self.layers)):
            d_e_W = - (d_ol_ol1 * np.expand_dims(self.d_ol_wl_mx(layer), axis=1))
            d_e_W = d_e_W.reshape((layer.sigmoid_ol.shape[0], self.layers[-1].sigmoid_ol.shape[1], -1))
            Jacobian = np.dstack((d_e_W, Jacobian))
            if ind != len(self.layers):
                d_ol_ol1 = d_ol_ol1.reshape((layer.sigmoid_ol.shape[0], self.layers[-1].sigmoid_ol.shape[1],
                                             layer.weights.shape[1])) @ self.d_ol_ol1_mx(layer)
                d_ol_ol1 = np.expand_dims(d_ol_ol1, axis=-1)

        np.set_printoptions(threshold=np.inf)
        return np.concatenate(Jacobian)

    def E_matrix(self, prediction_mx, target):
        E_mx = target - prediction_mx
        return E_mx

    def SSE(self, prediction_mx, target):
        E_mx = self.E_matrix(prediction_mx, target)
        E_square_mx = np.square(E_mx)
        SSE_vec = np.sum(E_square_mx, axis=1)
        return SSE_vec

    def Rec_Rate(self, prediction_mx, target):
        return np.sum(np.abs(np.argmax(prediction_mx, axis=1) - np.argmax(target, axis=1))) / prediction_mx.shape[0]

    def E_expabs_vector(self, tau, SSE_vector):
        return np.abs(tau) * np.exp(SSE_vector / np.abs(tau))

    def calculate_Error(self, inputs, targets, tau, mode=0):
        prediction_mx = self.forward_propagation(inputs, mode)
        SSE_vector = self.SSE(prediction_mx, targets)
        Eexpabs = np.sum(self.E_expabs_vector(tau, SSE_vector))

        output_number = targets.shape[1]

        RMSE = np.sqrt(np.mean(SSE_vector) / output_number)
        SSE = np.sum(SSE_vector) / 2

        RR = np.sum(np.argmax(prediction_mx, axis=1) == np.argmax(targets, axis=1)) / targets.shape[0]
        MSE = np.sum(SSE_vector) / (output_number * targets.shape[0])
        CE = np.sum(np.multiply(targets * (-1), np.log(prediction_mx))) / (output_number * targets.shape[0])

        return Eexpabs, SSE, RMSE, RR, MSE, CE

    def Gradient(self, AJ_mx, prediction_mx, target):
        return np.dot(np.transpose(AJ_mx), np.concatenate(self.E_matrix(prediction_mx, target)))

    def Hessian_SSE(self, Jacobian):
        return np.transpose(Jacobian) @ Jacobian

    @staticmethod
    def calculate_delta_weights_vector(mu, Hessian_ext, Gradient_ext):
        try:
            delta_Weights_vec = np.linalg.solve(Hessian_ext + np.identity(len(Hessian_ext)) * mu, Gradient_ext)
        except:
            delta_Weights_vec = np.dot(np.linalg.pinv(Hessian_ext + np.identity(len(Hessian_ext)) * mu), Gradient_ext)

        return delta_Weights_vec

    def update_weights_candidate(self, mu, Hessian_ext, Gradient_ext):
        delta_weights_vector = self.calculate_delta_weights_vector(mu, Hessian_ext, Gradient_ext)
        left_id = 0
        for layer in self.layers:
            right_id = left_id + layer.weights.shape[0] * layer.weights.shape[1]
            layer.weights_candidate = layer.weights - delta_weights_vector[left_id:right_id].reshape(
                (layer.weights.shape[0], layer.weights.shape[1]), order="F")
            left_id = right_id

    def MSE_iteration_fix(self, train_inputs, train_targets, val_inputs, val_targets, mu, ratio=10):
        m = 1

        train_prediction_mx = self.forward_propagation(train_inputs, mode=0)
        train_SSE_vector = self.SSE(train_prediction_mx, train_targets)

        train_MSE = np.sum(train_SSE_vector) / (train_targets.shape[1] * train_targets.shape[0])
        train_Ek = np.sum(self.E_expabs_vector(self.tau, train_SSE_vector))
        train_RR = np.sum(np.argmax(train_prediction_mx, axis=1) == np.argmax(train_targets, axis=1)) / \
                   train_targets.shape[0]
        train_CE = np.sum(np.multiply(train_targets * (-1), np.log(train_prediction_mx))) / (
                    train_targets.shape[1] * train_targets.shape[0])

        tr_Ek.append(train_Ek)
        tr_MSE.append(train_MSE)
        tr_RR.append(train_RR)
        tr_CE.append(train_CE)
        Tau.append(self.tau)
        Mu.append(mu)

        self.train_Error = train_MSE

        Jacobian = self.compute_Jacobian()
        Gradient = self.Gradient(Jacobian, train_prediction_mx, train_targets)
        Hessian = self.Hessian_SSE(Jacobian)

        self.update_weights_candidate(mu, Hessian, Gradient)

        train_Ekplus1, train_SSE, train_RMSE, train_RR, train_MSE_kplus1, train_CE = self.calculate_Error(train_inputs,
                                                                                                          train_targets,
                                                                                                          self.tau_candidate,
                                                                                                          mode=1)

        if train_MSE_kplus1 <= train_MSE:
            if mu >= 0.1 ** 12:
                mu = float(mu) / (float(ratio))

            gauss_grad.append(0)

        else:
            while ((train_MSE_kplus1 > train_MSE) and (m <= 5)):
                if mu < 1000:
                    mu = mu * ratio
                m += 1

                self.update_weights_candidate(mu, Hessian, Gradient)

                train_Ekplus1, train_SSE, train_RMSE, train_RR, train_MSE_kplus1, train_CE = self.calculate_Error(
                    train_inputs, train_targets, self.tau_candidate, mode=1)

            gauss_grad.append(1)

        for layer in self.layers:
            layer.weights = layer.weights_candidate

        val_Ekplus1, val_SSE, val_RMSE, val_RR, val_MSE, val_CE = self.calculate_Error(val_inputs, val_targets,
                                                                                       self.tau,
                                                                                       mode=1)
        vl_MSE.append(val_MSE)
        vl_RR.append(val_RR)
        vl_CE.append(val_CE)

        return val_MSE, mu, val_RR, val_Ekplus1, val_CE

    def fit(self, train_inputs, train_targets, val_inputs, val_targets, mu, initial_tau, max_iteration,
            early_max_stepsize=0, fix=False, alpha=0, eta=0, eta_plus=1.05, eta_minus=0.5, M_MAX=0,
            train_missing_input=None, MSE_training=False, err_delta=10e-7):

        iteration = 0
        early_iteration = 0

        early_Eexpabs, _, _, early_RR, early_MSE, early_CE = self.calculate_Error(val_inputs, val_targets, self.tau,
                                                                                  mode=1)
        vl_Eexpabs.append(early_Eexpabs)
        vl_MSE.append(early_MSE)
        vl_RR.append(early_RR)
        vl_CE.append(early_CE)

        early_Error = early_MSE

        self.optimal = OptimalValues(copy.deepcopy(self.layers), self.tau, initial_tau, fix, early_Eexpabs, early_MSE,
                                     early_CE, early_RR, mu, iteration)

        while (iteration < max_iteration) and (early_iteration < early_max_stepsize):
            iteration += 1
            early_iteration += 1
            if fix == False:
                raise NotImplementedError("Only fixed MSE training is implemented!")

            elif fix == True:
                if train_missing_input is None:
                    if MSE_training != True:
                        raise NotImplementedError("Only MSE training is implemented!")
                    else:
                        val_Ek, mu, val_RR, val_Eexpabs, val_CE = self.MSE_iteration_fix(train_inputs, train_targets,
                                                                                         val_inputs, val_targets, mu,
                                                                                         ratio=10)
                        vl_MSE.append(val_Ek)
                else:
                    raise NotImplementedError("Only MSE training is implemented!")

            if early_Error > (val_Ek + err_delta) :
                early_Error = val_Ek
                self.optimal = OptimalValues(copy.deepcopy(self.layers), self.tau, initial_tau, fix, val_Ek, val_Ek,
                                             val_CE, val_RR, mu, iteration)
                early_iteration = 0
            else:
                pass

        end_tr_Error, _, _, end_tr_RR, end_tr_MSE, end_tr_CE = self.calculate_Error(train_inputs, train_targets,
                                                                                    self.tau, mode=1)
        tr_Ek.append(end_tr_Error)
        tr_MSE.append(end_tr_MSE)
        tr_RR.append(end_tr_RR)
        tr_CE.append(end_tr_CE)
        gauss_grad.append(np.nan)
        Mu.append(mu)
        Eta.append(eta)

    def evaluate(self, test_inputs):
        return self.forward_propagation(test_inputs, mode=2)