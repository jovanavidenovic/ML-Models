import os
import numpy as np
import math
import random


class PCA:
    def __init__(
        self,
        n_components: int,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        rnd_seed: int = 0,
    ):
        assert (
            type(n_components) == int
        ), f"n_components is not of type int, but {type(n_components)}"
        assert (
            n_components > 0
        ), f"n_components has to be greater than 0, but found {n_components}"

        self.n_components = n_components
        self.dualpca_eigenvectors = []
        self.eigenvectors = []
        self.eigenvalues = []
        self.mean = 0

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rnd_seed = rnd_seed

    def fit(self, X: np.ndarray) -> None:
        """
        Fit principle component vectors.
        Center the data around zero.

        Arguments
        ---------
        X: np.ndarray
            Data matrix with shape (n_samples, n_features)
        """
        X_copy = X.copy()
        self.mean = np.mean(X, axis=0)
        X_copy = (X_copy.T - self.mean.reshape(len(self.mean), 1)).T

        if(X.shape[0] > X.shape[1]):
            vector = np.ones((X_copy.shape[1], 1))
        else:
            vector = np.ones((X_copy.shape[0], 1))
        
        vector = vector / np.linalg.norm(vector)
        if(X.shape[0] > X.shape[1]):
            M = np.cov(X_copy.T)
        else:
            M = np.cov(X_copy)
        for _ in range(self.n_components):
            eigvector, eigvalue = self.power_method(M, vector, 0)
            self.eigenvalues.append(eigvalue)
            self.dualpca_eigenvectors.append(eigvector)
            M = M - eigvalue * eigvector * np.array(eigvector).T   
            if(X.shape[0] <= X.shape[1]):
                eigvector = np.dot(X_copy.T, eigvector) #/ np.sqrt(X.shape[0] * eigvalue)
                eigvector /= np.linalg.norm(eigvector)
            self.eigenvectors.extend(eigvector.reshape(1, len(eigvector)))
        self.eigenvectors = np.array(self.eigenvectors)

    def power_method(
        self, M: np.ndarray, vector: np.array, iteration: int = 0
    ) -> tuple:
        """
        Perform the power method for calculating the eigenvector with the highest corresponding
        eigenvalue of the covariance matrix.
        This should be a recursive function. Use 'max_iterations' and 'tolerance' to terminate
        recursive call when necessary.

        Arguments
        ---------
        M: np.ndarray
            Covariance matrix of the zero centered data.
        vector: np.array
            Candidate eigenvector in the iteration.
        iteration: int
            Index of the consecutive iteration for termination purpose of the

        Return
        ------
        np.array
            The unit eigenvector of the covariance matrix.
        float
            The corresponding eigenvalue of the covariance matrix.
        """
        vector_new = np.dot(M, vector)
        vector_new /= np.linalg.norm(vector_new)
        if(iteration >= self.max_iterations or sum(abs(vector_new - vector)) <= self.tolerance):
            eigenvalue = (vector_new.T @  M) @ vector_new 
            eigenvalue = eigenvalue[0][0]
            return (vector_new, eigenvalue)
        else:
            return self.power_method(M, vector_new, iteration + 1)        

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data (X) using fitted eigenvectors

        Arguments
        ---------
        X: np.ndarray
            New data with the same number of features as the fitting data.

        Return
        ------
        np.ndarray
            Transformed data with the shape (n_samples, n_components).
        """
        eigenvecs = self.eigenvectors.T
        X_copy = X.copy().T - self.mean.reshape(len(self.mean), 1)
        n_eigenvectors = eigenvecs[:, 0:self.n_components]
        transformed_X = np.dot(n_eigenvectors.T, X_copy)
        return transformed_X.T

    def get_explained_variance(self):
        """
        Return the explained variance ratio of the principle components.
        Prior to calling fit() function return None.
        Return only the ratio for the top 'n_components'.

        Return
        ------
        np.array
            Explained variance for the top 'n_components'.
        """
        if(len(self.eigenvalues) == 0):
            return None
        return self.eigenvalues[:self.n_components] 

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data from the principle component space into
        the real space.

        Arguments
        ---------
        X: np.ndarray
            Data  in PC space with the same number of features as
            the fitting data.

        Return
        ------
        np.ndarray
            Transformed data in original space with
            the shape (n_samples, n_components).
        """
        eigenvecs = self.eigenvectors.T
        n_eigenvectors = eigenvecs[:, 0:self.n_components]
        transformed_X = np.dot(n_eigenvectors, X.T) + self.mean.reshape(len(self.mean), 1)
        return transformed_X.T
