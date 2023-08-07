from typing import Tuple
import numpy as np

from ..interfaces import IModel, IFeatureEngineering


class MultipleLinearRegression(IModel):
    """
    Multiple Feature Linear Regression Model

    Advantages
    ----------
    - Easy to implement
    - Easy to interpret the output
    - Computationally cheap

    Disadvantages
    -------------
    - Poor performance on non-linear data
    - Sensitive to outliers
    - Sensitive to overfitting
    """

    def __init__(
        self,
        learning_rate: np.float64 = 0.0001,
        num_iterations: int = 10000,
        lambda_: np.float64 = 0,
        debug: bool = True,
        copy_X: bool = True,
        X_scalar: IFeatureEngineering = None,
        Y_scalar: IFeatureEngineering = None,
    ) -> None:
        """
        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate, by default 0.0001
        num_iterations : int, optional
            The number of iterations, by default 10000
        lambda_ : np.float64, optional
            The regularization parameter, by default 0
        debug : bool, optional
            Whether to print debug messages, by default True
        copy_X : bool, optional
            Whether to copy the input array, by default True
        X_scalar : IFeatureEngineering, optional
            The feature scaling object for the input array, by default None
        Y_scalar : IFeatureEngineering, optional
            The feature scaling object for the output array, by default None
        """
        self._learning_rate = learning_rate
        self._num_iterations = num_iterations
        self._lambda_ = lambda_
        self._debug = debug
        self._copy_X = copy_X
        self._X_scalar = X_scalar
        self._Y_scalar = Y_scalar

        self._weights: np.ndarray = None
        self._intercept: np.float64 = None

        self._cost_history: np.ndarray = None
        self._params_history: np.ndarray = None

        self._debug_freq = num_iterations // 10

    def _y_hat(self, X: np.ndarray, W: np.ndarray, b: np.float64) -> np.float64:
        """
        Return the predicted value of y given x, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_features,)
        W : np.ndarray
            The weight array of shape (n_features,)
        b : np.float64
            The intercept

        Returns
        -------
        np.float64
            The predicted value of y
        """
        return np.dot(X, W) + b

    def _cost(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        Return the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        Y : np.ndarray
            The output array of shape (n_samples,)
        W : np.ndarray
            The weight array of shape (n_features,)
        b : np.float64
            The intercept

        Returns
        -------
        np.float64
            The computed cost
        """
        """
        ALTERNATIVE IMPLEMENTATION
        --------------------------
        m = X.shape[0]
        cost = 0.0

        for i in range(m):
            cost += (self._y_hat(X[i], W, b) - Y[i]) ** 2

        return cost / (2 * m)
        """
        m = X.shape[0]
        return np.sum((self._y_hat(X, W, b) - Y) ** 2) / (2 * m)

    def _gradient(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.float64
    ) -> Tuple[np.ndarray, np.float64]:
        """
        Return the gradient of the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        Y : np.ndarray
            The output array of shape (n_samples,)
        W : np.ndarray
            The weight array of shape (n_features,)
        b : np.float64
            The intercept

        Returns
        -------
        Tuple[np.ndarray, np.float64]
            The computed gradient
        """

        """
        ALTERNATIVE IMPLEMENTATION
        --------------------------
        m, n = X.shape

        dw = np.zeros((n,))
        db = 0.0

        for i in range(m):
            dw += (self._y_hat(X[i], W, b) - Y[i]) * X[i]
            db += self._y_hat(X[i], W, b) - Y[i]

        return dw / m, db / m
        """

        m = X.shape[0]
        dw = np.dot(X.T, self._y_hat(X, W, b) - Y) / m
        db = np.sum(self._y_hat(X, W, b) - Y) / m
        return dw, db

    def _getXandY(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the input and output arrays.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        Y : np.ndarray
            The output array of shape (n_samples,) or (n_samples, 1)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The input and output arrays
        """
        if self._copy_X:
            X = np.copy(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        if self._X_scalar is not None:
            X = self._X_scalar.transform(X)

        if self._Y_scalar is not None:
            Y = self._Y_scalar.transform(Y).reshape(-1)
        else:
            Y = Y.reshape(-1)

        return X, Y

    def fit(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0
    ) -> None:
        """
        Train the model given X and Y.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features) or (n_samples,)
        Y : np.ndarray
            The output array of shape (n_samples,) or (n_samples, 1)
        w : np.ndarray, optional
            The weight array, by default None
        b : np.float64, optional
            The intercept, by default 0

        Returns
        -------
        None
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

        X, Y = self._getXandY(X, Y)

        self._weights = np.zeros(X.shape[1]) if W is None else W
        self._intercept = b

        self._cost_history = [self._cost(X, Y, self._weights, self._intercept)]
        self._params_history = []

        for i in range(self._num_iterations):
            dw, db = self._gradient(X, Y, self._weights, self._intercept)

            self._weights = self._weights.astype("float64")
            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            self._cost_history.append(self._cost(X, Y, self._weights, self._intercept))
            self._params_history.append((self._weights, self._intercept))

            if self._debug and i % self._debug_freq == 0:
                self._printIteration(i)

        if self._debug:
            self._printIteration(self._num_iterations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted values given X.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features) or (n_samples,)

        Returns
        -------
        np.ndarray
            The predicted values of shape (n_samples,)
        """
        assert self._weights is not None and self._intercept is not None, (
            "The model must be trained before making predictions. "
            "Call the fit method first."
        )

        X, _ = self._getXandY(X, np.zeros(X.shape[0]))

        predictions = [
            self._y_hat(X[i], self._weights, self._intercept) for i in range(X.shape[0])
        ]

        if self._Y_scalar is not None:
            predictions = self._Y_scalar.inverse_transform(predictions)

        return np.array(predictions)

    def get_cost_history(self) -> np.ndarray:
        """
        Return the history of the cost function.

        Returns
        -------
        np.ndarray
            The history of the cost function
        """
        return np.array(self._cost_history)

    def get_parameter_history(self) -> Tuple[np.ndarray, np.float64]:
        """
        Return the history of the parameters.

        Returns
        -------
        Tuple[np.ndarray, np.float64]
            The history of the parameters
        """
        return np.array(self._params_history)

    def get_weights(self) -> np.ndarray:
        """
        Return the weights.

        Returns
        -------
        np.ndarray
            The weights
        """
        return self._weights

    def get_intercept(self) -> np.float64:
        """
        Return the intercept.

        Returns
        -------
        np.float64
            The intercept
        """
        return self._intercept

    def score(
        self, X: np.ndarray, Y: np.ndarray, w: np.ndarray = None, b: np.float64 = None
    ) -> np.float64:
        """
        Return the cost for given X and Y.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features) or (n_samples,)
        Y : np.ndarray
            The output array of shape (n_samples,) or (n_samples, 1)
        w : np.ndarray, optional
            The weight array, by default None
        b : np.float64, optional
            The intercept, by default None

        Returns
        -------
        np.float64
            The computed cost
        """

        X, Y = self._getXandY(X, Y)

        if w is None:
            w = self._weights

        if b is None:
            b = self._intercept

        return self._cost(X, Y, w, b)

    def _printIteration(self, iteration: int) -> None:
        """
        Print the current iteration and cost.

        Parameters
        ----------
        iteration : int
            The current iteration
        """
        n = len(str(self._num_iterations)) + 1
        cost = self._cost_history[-1]
        print(f"Iteration: {iteration:{n}n} | Cost: {cost:0.6e}")
