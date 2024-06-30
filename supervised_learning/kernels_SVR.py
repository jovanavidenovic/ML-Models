import math
import random
import cvxopt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR as SVR_sk
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

RANDOM_SEED = 1
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Linear:
    """An example of a kernel."""

    def __init__(self):
        # here a kernel could set its parameters
        pass

    def name(self):
        return "Linear"
    
    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)

class Polynomial:
    def __init__(self, M):
        self.M = M

    def name(self):
        return f"Poly({self.M})"

    def __call__(self, x, x_):
        return (1 + x.dot(x_.T)) ** self.M

class RBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def name(self):
        return f"RBF({self.sigma})"

    def __call__(self, X, X_):
        if len(X.shape) == 1 and len(X_.shape) == 1:
            return np.exp(-np.linalg.norm(X - X_) ** 2 / (2 * self.sigma ** 2))

        if len(X.shape) == 1 and len(X_.shape) == 2 or len(X.shape) == 2 and len(X_.shape) == 1:
            X_l = X if len(X.shape) == 1 else X_
            X_ = X_ if len(X.shape) == 1 else X
            X = X_l

            X = X.reshape(1, -1)
            diag_X_ = np.diag(X_.dot(X_.T))
            power = np.sum(X ** 2) - 2 * X.dot(X_.T) + diag_X_
            power = power.reshape(-1)
            return np.exp(-power / (2 * self.sigma ** 2))

        if len(X.shape) == 2 and len(X_.shape) == 2:
            diag_X = np.diag(X.dot(X.T))
            diag_X_ = np.diag(X_.dot(X_.T))

            diag_X = np.repeat(diag_X.reshape(-1, 1), X_.shape[0], axis=1)
            diag_X_ = np.repeat(diag_X_.reshape(1, -1), X.shape[0], axis=0)

            power = diag_X- 2 * X.dot(X_.T) + diag_X_
            return np.exp(-power / (2 * self.sigma ** 2))

        raise ValueError("Invalid input shapes")    
            
class KernelizedRidgeRegression():
    def __init__(self, kernel, lambda_, std=False):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.std = std

    def name(self):
        return f"KRR ({self.kernel.name()}, λ={self.lambda_})"
    
    def fit(self, X, y):
        if self.std:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        else:
            self.X = X
        K = self.kernel(self.X, self.X)
        self.betas = np.linalg.inv(K + self.lambda_ * np.eye(K.shape[1])).dot(y)
        return self

    def predict(self, X):
        if self.std:
            X = self.scaler.transform(X)
        k = self.kernel(X, self.X)
        return k.dot(self.betas)

class SVR:
    def __init__(self, kernel, lambda_, epsilon, std=False):
        self.kernel = kernel
        self.C = 1 / lambda_
        self.eps = epsilon
        self.std = std  
    
    def name(self):
        return f"SVR ({self.kernel.name()}, C={self.C}, ε={self.eps})"

    def get_alpha(self):
        alphas = np.hstack((self.alphas, self.alphas_star))
        return alphas

    def get_b(self):
        return self.bias     

    def fit(self, X, y):
        if self.std:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        else:
            self.X = X

        l = len(X)
        odd_idx = np.arange(1, 2 * l, 2)
        K = self.kernel(self.X, self.X)

        A = np.ones((2 * l, 1))
        A[odd_idx] = -1
        A = A.T

        b = np.array([[0.0]]).reshape(-1, 1)

        G_upper = np.eye(2 * l)
        G_lower = -1 * np.eye(2 * l)
        G = np.vstack((G_upper, G_lower))

        h_upper = self.C * np.ones((2 * l, 1))
        h_lower = np.zeros((2 * l, 1))
        h = np.vstack((h_upper, h_lower))

        q_right = np.repeat(y, 2)
        q_right[odd_idx] *= -1
        q_left = self.eps * np.ones((2 * l))
        q = q_left - q_right
        q = q.reshape(-1, 1)

        P = np.kron(K, np.array([[1, -1], [-1, 1]])) 

        A = cvxopt.matrix(A)
        G = cvxopt.matrix(G)
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        h = cvxopt.matrix(h)
        b = cvxopt.matrix(b)

        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas_all = np.array(sol["x"]).reshape(-1)
        self.alphas = alphas_all[::2].reshape(-1, 1)
        self.alphas_star = alphas_all[1::2].reshape(-1, 1)

        self.bias = sol["y"][0]
        
        return self

    def predict(self, X):
        if self.std:
            X = self.scaler.transform(X)
        k = self.kernel(X, self.X)
        pred =  k.dot(self.alphas - self.alphas_star) + self.bias
        return pred.reshape(-1)

    def is_sv(self):
        svs = np.abs(self.alphas - self.alphas_star) > 1e-3
        return svs
