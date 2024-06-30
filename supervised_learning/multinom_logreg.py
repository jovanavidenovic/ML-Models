import math
import scipy
import random
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 4
MBOG_TRAIN = 100
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def std_err(x):
    mi = np.mean(x)
    m = len(x)
    var_l  = 1/(m-1)*sum((x-mi)**2)
    return math.sqrt(var_l/m)    

def dataset_preps(X, y):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_1he = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray().astype(int)
    return X, y_1he

def get_multinom_logl(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    p = p / p.sum(axis=1)[:, np.newaxis]
    return -np.log(p[np.arange(len(y)), y])

def get_01_loss(y, pred_y):
    return y != pred_y

class MajorityClassifier:
    def build(self, X, y):
        class_counts = np.bincount(y)
        ref_class = np.argmax(class_counts)
        return MajorityClassifierModel(ref_class, len(class_counts))

class MajorityClassifierModel:
    def __init__(self, ref_class, num_classes):
        self.ref_class = ref_class
        self.num_classes = num_classes

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.num_classes))
        probs[:, self.ref_class] = 1
        return probs

class MultinomialLogReg:
    def __logl(self, params, X, y_1he, ref_class_idx):
        num_samples, num_features = X.shape
        num_classes = y_1he.shape[1]

        params = params.reshape((num_classes-1, num_features))

        params = np.vstack((params[:ref_class_idx,], np.zeros((num_features, 1)).T, params[ref_class_idx:, ]))

        u = np.dot(params, X.T)
        p = np.exp(u) / np.sum(np.exp(u), axis=0)

        p = np.clip(p, 1e-15, 1 - 1e-15)
        p = p / p.sum(axis=0)

        logl = np.sum(np.log(p[y_1he.argmax(axis=1), np.arange(num_samples)]))
        return -logl

    def build(self, X, y, ref_class_idx=0):
        X, y_1he = dataset_preps(X, y)
        num_features = X.shape[1]
        num_classes = len(np.unique(y))

        params = np.zeros((num_classes-1, num_features))
        params = scipy.optimize.fmin_l_bfgs_b(self.__logl, params, args=(X, y_1he, ref_class_idx), approx_grad=True)[0]

        params = params.reshape((num_classes-1, num_features))
        params = np.vstack((params[:ref_class_idx,], np.zeros((num_features, 1)).T, params[ref_class_idx:, ]))
        return MNLRClassifier(params)
        
class MNLRClassifier:
    def __init__(self, params):
        self.params = params

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        u = np.dot(self.params, X.T)
        p = np.exp(u) / np.sum(np.exp(u), axis=0)
        return p.T

class OrdinalLogReg:
    def __logl(self, params, X, y):
        _, num_features = X.shape
        
        betas, deltas = params[:num_features], params[num_features:]
        thrs = np.cumsum(deltas)
        thrs = np.concatenate((np.array([-np.inf, 0]), thrs, np.array([np.inf])))
        thrs = thrs.reshape(-1, 1)

        u = np.dot(betas, X.T)
        u = u.reshape(-1, 1)
        
        F_j_1 = 1 / (1 + np.exp(-(thrs[y + 1] - u)))
        F_j = 1 / (1 + np.exp(-(thrs[y] - u)))
        logl = np.sum(np.log(F_j_1 - F_j))
        return -logl

    def build(self, X, y):
        X, _ = dataset_preps(X, y)
        num_features = X.shape[1]
        num_classes = len(np.unique(y))

        betas = np.ones(num_features)
        deltas = np.ones(num_classes-2) * 1e-2

        bounds = [(None, None)] * num_features + [(1e-6, None)] * (num_classes-2)

        params = np.hstack((betas, deltas))
        params = scipy.optimize.fmin_l_bfgs_b(self.__logl, params, args=(X, y), approx_grad=True, bounds=bounds)[0]

        betas, deltas = params[:num_features], params[num_features:]
        return OLRClassifier(betas, deltas)
    
class OLRClassifier:
    def __init__(self, betas, deltas):
        self.betas = betas
        self.thrs = np.cumsum(deltas)
        self.thrs = np.concatenate((np.array([-np.inf, 0]), self.thrs, np.array([np.inf])))
        self.thrs = self.thrs.reshape(-1, 1)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        u = np.dot(self.betas, X.T)
        p = 1 / (1 + np.exp(-(self.thrs[1:] - u))) - 1 / (1 + np.exp(-(self.thrs[:-1] - u)))
        return p.T
