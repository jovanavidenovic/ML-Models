import numpy as np


def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """A cost function in matrix/vector form.
    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters
    Returns
    -------
    float
    """
    m = len(y)
    f_X = np.dot(X, theta)
    errors = (f_X - y)**2
    cost = sum(errors)/(2*m)
    return cost


def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient of cost function in matrix/vector form.
    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters
    Returns
    -------
    numpy array of shape (n_features,)
    """
    m = len(y)
    f_X = np.dot(X, theta)
    return 1/m * np.dot(X.T, f_X - y)       

def gradient_descent(
    X: np.ndarray, y: np.ndarray, lr=0.01, tol=1e-7, max_iter=10_000
) -> np.ndarray:
    """Implementation of gradient descent.
    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    lr: float
        The learning rate.
    tol: float
        The stopping criterion (tolerance).
    max_iter: int
        The maximum number of passes (aka epochs).
    Returns
    -------
    numpy array of shape (n_features,)
    """
    _, n = X.shape
    theta = np.ones((n, ))
    iter_idx = 0
    loss = cost(X, y, theta)
    while(iter_idx < max_iter and loss > tol):
        grad = gradient(X, y, theta)
        theta = theta - lr*grad
        iter_idx += 1
        loss = cost(X, y, theta)
    return theta
    
class LinearRegression:
    def __int__(self) -> None:
        self.coefs: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        The fit method of LinearRegression accepts X and y
        as input and save the coefficients of the linear model.
        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
            Training data
        y: numpy array of shape (n_samples,)
            Target values
        Returns
        -------
        None
        """
        m = len(y)
        X_new = np.insert(X, 0, np.ones((m, )), 1)
        coefs = gradient_descent(X_new, y)
        self.coefs = coefs[1:]
        self.intercept = coefs[0]
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.
        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
            New samples
        Returns
        -------
        numpy array of shape (n_samples,)
            Returns predicted values.
        """
        m, _ = X.shape
        X_new = np.insert(X, [0], np.ones((m, 1)), 1)
        coefs_merged = np.insert(self.coefs, 0, self.intercept)
        return np.dot(X_new, coefs_merged)
