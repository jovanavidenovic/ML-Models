import math
import torch
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

EPS = 1e-7
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class EarlyStopper:
    def __init__(self, patience=50, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')
        self.hard_patience = 5000
        self.best_weights = None

    def early_stop(self, validation_loss, weights):
        if self.hard_patience == 0:
            print("Hard patience reached")
            return True
        
        if validation_loss >= self.min_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            self.hard_patience -= 1
        else:
            if validation_loss < self.min_loss:
                self.min_loss = validation_loss
                self.hard_patience = 5000
                self.best_weights = weights
            else:
               self.hard_patience -= 1 

            self.counter = 0
        return False

class Layer_FullyConnected():
    def __init__(self, num_inputs, num_neurons, lambda_=0.0001):
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.past_weights = np.zeros((num_inputs, num_neurons))
        self.biases = np.zeros((1, num_neurons))
        self.past_biases = np.zeros((1, num_neurons))
        self.lambda_ = lambda_

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        return self.outputs
    
    def backprop(self, local_grad):
        self.dweights = np.dot(self.inputs.T, local_grad) + self.lambda_ * self.weights
        self.dbiases = np.sum(local_grad, axis=0, keepdims=True)
        der = np.dot(local_grad, self.weights.T)
        return der

    def update_params(self, lr, momentum=0.9):
        tmp_weights = self.weights
        tmp_biases = self.biases
        self.weights = self.weights - lr * self.dweights / len(self.inputs) + momentum * (self.weights - self.past_weights)
        self.biases = self.biases - lr * self.dbiases / len(self.inputs) + momentum * (self.biases - self.past_biases)
        self.past_weights = tmp_weights
        self.past_biases = tmp_biases    
        
class Activation_ReLU():
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backprop(self, local_grad):
        local_grad[self.outputs <= 0] = 0
        return local_grad

class Layer_Softmax():
    def __init__(self, eps=1e-7):
        self.eps = eps

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # exp_values = np.exp(inputs)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.outputs

    def backprop(self, local_grad):
        return local_grad

class Loss:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        data_loss = np.sum(sample_losses, axis=0)
        return data_loss

class Loss_MSE(Loss):
    def forward(self, y_pred, y_true):
        self.ypred = y_pred
        self.ytrue = y_true
        return (y_true - y_pred) ** 2
    
    def backprop(self):
        return -2 * (self.ytrue - self.ypred)

class Loss_CCE(Loss):
    def forward(self, y_pred, y_true):
        self.ypred = np.clip(y_pred, 1e-200, 1 - 1e-200)
        self.ytrue = y_true
        res = -np.log(np.sum(self.ypred * self.ytrue, axis=1))
        return res
    
    def backprop(self):
        return self.ypred - self.ytrue

class ANN():
    def __init__(self, units, lambda_, name=""):
        self.units = units
        self.lambda_ = lambda_
        self.layers = None
        self.name = name

    def _create_network(self, X, y):
        self.layers = [Layer_FullyConnected(X.shape[1], self.units[0])]
        self.layers.append(Activation_ReLU())
        for i in range(1, len(self.units)):
            self.layers.append(Layer_FullyConnected(self.units[i-1], self.units[i]))
            self.layers.append(Activation_ReLU())

    def _forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def _backpropagation(self, local_grad):
        for layer in reversed(self.layers):
            local_grad = layer.backprop(local_grad)
        return local_grad
    
    def __pass_batch(self, X, y, lr):
        outputs = self._forward(X)
        weights_loss_part = np.array(self.weights(no_bias=True))
        weights_loss_part = np.sum(weights_loss_part**2)
        loss = self.loss_function.forward(outputs, y)
        loss = np.sum(loss) + self.lambda_ * weights_loss_part
        local_grad = self.loss_function.backprop()
        self._backpropagation(local_grad)
        for layer in self.layers:
            if isinstance(layer, Layer_FullyConnected):
                layer.update_params(lr)
        return loss            

    def _train_network(self, X, y, lr=1e-2, epochs=5000, val=None):
        batch_size = 64
        if val:
            X_val, y_val, patience, min_delta = val
            early_stopper = EarlyStopper(patience, min_delta)

        for epoch in range(epochs):
            X_shuffled, y_shuffled = X, y
            perm = np.random.permutation(X_shuffled.shape[0])
            X_shuffled = X_shuffled[perm]
            y_shuffled = y_shuffled[perm]
            total_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                loss = self.__pass_batch(X_batch, y_batch, lr)
                total_loss += loss

            total_loss /= X.shape[0]
            if val:
                outputs = self._forward(X_val)
                val_loss = self.loss_function.calculate(outputs, y_val)
                val_loss = val_loss / X_val.shape[0]
                if epoch > 100:
                    stop = early_stopper.early_stop(val_loss, self.weights())
                    if stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
    def predict(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer.forward(outputs)
        outputs = outputs.reshape(-1) if outputs.shape[1] == 1 else outputs
        return outputs

    def weights(self, no_bias=False):
        # a list of weight matrices that include intercept biases
        W = []
        for layer in self.layers:
            if isinstance(layer, Layer_FullyConnected):
                if no_bias:
                    W.extend(layer.weights.reshape(-1))
                else:
                    # join weights and biases
                    W.append(np.concatenate((layer.weights, layer.biases), axis=0))
        return W
    
    def _get_gradients(self):
        grads = []
        for layer in self.layers:
            if isinstance(layer, Layer_FullyConnected):
                grads.append(np.concatenate((layer.dweights, layer.dbiases), axis=0))
        return grads
    
    def _set_weights(self, weights):
        layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, Layer_FullyConnected):
                layer.weights = weights[layer_idx][:-1]
                layer.biases = weights[layer_idx][-1].reshape(1, -1)
                layer_idx += 1

class ANNClassification(ANN):
    def __init__(self, units, lambda_, name=""):
        super().__init__(units, lambda_, name)
        self.loss_function = Loss_CCE(self.lambda_)

    def fit(self, X, y, lr=1e-1, epochs=5000, val=None):
        y_1he = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        self.layers = []
        if len(self.units) > 0:
            super()._create_network(X, y)
        last_layer_input_size = self.units[-1] if len(self.units) > 0 else X.shape[1]
        self.layers.append(Layer_FullyConnected(last_layer_input_size, len(np.unique(y))))
        self.layers.append(Layer_Softmax())
        super()._train_network(X, y_1he, lr=lr, epochs=epochs, val=val)
        return self

class ANNRegression(ANN):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_)
        self.loss_function = Loss_MSE(self.lambda_)

    def fit(self, X, y, lr=1e-1, epochs=5000, val=None):
        y = y.reshape(-1, 1)
        self.layers = []
        if len(self.units) > 0:
            super()._create_network(X, y)
        last_layer_input_size = self.units[-1] if len(self.units) > 0 else X.shape[1]
        self.layers.append(Layer_FullyConnected(last_layer_input_size, 1))
        super()._train_network(X, y, lr=lr, epochs=epochs, val=val)
        return self
