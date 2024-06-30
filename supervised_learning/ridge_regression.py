import numpy as np

class LinearRegression:
    def __init__(self, l2_regularization_coefficient: float = 0):
        self.l2_lambda = l2_regularization_coefficient
        self.coefs = None
        self.intercept = None

    def fit(self, X: np.ndarray, y: np.array) -> None:
        """
        Implement a fit method that uses a closed form solution to calculate coefficients and the intercept.
        Use vectorized operations to calculate weights. Do not forget about the intercept.
        Assign coefficients (np.array) to the self.coefs variable and intercept (float) to the self.intercept variable.

        Arguments
        ---------
        X: np.ndarray
            Array of features with rows as samples and features as columns.
        y: np.array
            Array of response variables with length of samples.
        """
        m, _ = X.shape
        X_new = np.insert(X, [0], np.ones((m, 1)), 1)
        _, n = X_new.shape
        modified_I = np.eye(n)
        modified_I[0][0] = 0

        w = np.linalg.inv(X_new.T @ X_new + self.l2_lambda*modified_I) @ X_new.T @ y    
        self.intercept = w[0]
        self.coefs = w[1:]

    def predict(self, X: np.ndarray) -> np.array:
        """
        Make prediction using learned coefficients and the intercept.
        Use vectorized operations.

        Arguments
        ---------
        X: np.ndarray
            Test data with rows as samples and columns as features

        Returns
        -------
        np.array
            Predicted values for given samples.
        """
        m, _ = X.shape
        coefs_merged = np.insert(self.coefs, 0, self.intercept)
        X_new = np.insert(X, [0], np.ones((m, 1)), 1)
        return X_new @ coefs_merged

def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray, l2_lambda: float) -> float:
    """
    A cost function in matrix/vector form. Stick to the notation from instructions to
    keep the lambda parameter equivalent between implementations.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters
    l2_lambda: float
        L2 regularization parameter

    Returns
    -------
    float
        The value of the cost function
    """
    m = len(y)
    f_X = np.dot(X, theta)
    errors = (f_X - y)**2
    cost = sum(errors)/2
    regularization_term = 1/2*l2_lambda*sum([theta[t_idx]**2 for t_idx in range(len(theta)) if t_idx != 0])
    return cost + regularization_term

def gradient(
    X: np.ndarray, y: np.ndarray, theta: np.ndarray, l2_lambda: float
) -> np.ndarray:
    """Gradient of cost function in matrix/vector form.
    Stick to the notation from instructions to
    keep the lambda parameter equivalent between implementations.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters
    l2_lambda: float
        L2 regularization parameter

    Returns
    -------
    numpy array of shape (n_features,)
    """
    m = len(y)
    f_X = np.dot(X, theta)
    theta_changed = theta.copy()
    theta_changed[0] = 0
    return np.dot(X.T, f_X - y) + l2_lambda*theta_changed       

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr=0.005,
    l2_lambda: float = 0.0,
    tol=1e-11,
    max_iter=100000,
):
    """Implementation of gradient descent.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    lr: float
        The learning rate.
    l2_lambda: float
        L2 regularization parameter.
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
    loss = cost(X, y, theta, l2_lambda)
    while(iter_idx < max_iter and loss > tol):
        grad = gradient(X, y, theta, l2_lambda)
        theta = theta - lr*grad
        iter_idx += 1
        loss = cost(X, y, theta, l2_lambda)
    return theta

class LinearRegressionGD:
    def __init__(self, l2_regularization_coefficient: float = 0) -> None:
        self.coefs = None
        self.intercept = None
        self.l2_lambda = l2_regularization_coefficient

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
        coefs = gradient_descent(X_new, y, l2_lambda=self.l2_lambda)
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
        X_new = np.insert(X, 0, np.ones((m, )), 1)
        coefs_merged = np.insert(self.coefs, 0, self.intercept)
        return np.dot(X_new, coefs_merged)    

def find_best_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    candidate_lambdas: np.ndarray = np.array([0, 0.01, 1, 10]),
):
    """
    Find the optimal L2 lambda parameter on train and validation dataset.
    For each lambda, instantiate a new LinearRegression model with that parameter,
    train it on the train set and evaluate performance with MSE on the validation set.
    Return lambda with the best MSE on the validation set.

    Arguments
    ---------
    X_train: np.ndarray
        Train dataset with rows as samples and columns as features.
    y_train: np.ndarray
        Train target vector with length of validation samples.
    X_validation: np.ndarray
        Validation dataset with rows as samples and columns as features.
    y_validation: np.ndarray
        Validation target vector with length of validation samples.
    candidate_lambdas: np.ndarray
        A list of float values for L2 regularization parameter (lambda).

    Returns
    -------
    float
        The best lambda from candidates according to MSE on validation set.
    """
    best_lambda = None
    min_mse = None
    for cand_lambda in candidate_lambdas:
        lr = LinearRegression(cand_lambda)
        lr.fit(X_train, y_train)
        validation_predict = lr.predict(X_validation)
        curr_mse = np.mean((y_validation - validation_predict) ** 2)
        if(best_lambda == None or curr_mse < min_mse):
            min_mse = curr_mse
            best_lambda = cand_lambda
    return best_lambda


