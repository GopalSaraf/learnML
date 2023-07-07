from typing import Tuple
import numpy as np
import copy

from ..interfaces import IModel
from ..interfaces import IFeatureScaling


class MultipleLinearRegression(IModel):
    """Multiple Feature Linear Regression Model"""

    def __init__(
        self,
        learning_rate: np.float64 = 0.0001,
        num_iterations: int = 10000,
        debug: bool = True,
        copy_X: bool = True,
        X_scalar: IFeatureScaling = None,
        Y_scalar: IFeatureScaling = None,
    ) -> None:
        """
        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate, by default 0.0001
        num_iterations : int, optional
            The number of iterations, by default 10000
        debug : bool, optional
            Whether to print debug messages, by default True
        copy_X : bool, optional
            Whether to copy the input array, by default True
        X_scalar : IFeatureScaling, optional
            The feature scaling object for the input array, by default None
        Y_scalar : IFeatureScaling, optional
            The feature scaling object for the output array, by default None
        """
        super().__init__(
            learning_rate, num_iterations, X_scalar, Y_scalar, debug, copy_X
        )

        self._weights: np.ndarray = None
        self._intercept: np.float64 = None

    def _y_hat(self, x: np.ndarray, w: np.ndarray, b: np.float64) -> np.float64:
        """
        Return the predicted value of y given x, w, and b.

        Parameters
        ----------
        x : np.ndarray
            The input array
        w : np.ndarray
            The weight array
        b : np.float64
            The intercept

        Returns
        -------
        np.float64
            The predicted value of y
        """
        return np.dot(x, w) + b

    def _cost(
        self, X: np.ndarray, Y: np.ndarray, w: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        Return the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : np.ndarray
            The weight array
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
            cost += (self._y_hat(X[i], w, b) - Y[i]) ** 2

        return cost / (2 * m)
        """
        m = X.shape[0]
        return np.sum((self._y_hat(X, w, b) - Y) ** 2) / (2 * m)

    def _gradient(
        self, X: np.ndarray, Y: np.ndarray, w: np.ndarray, b: np.float64
    ) -> Tuple[np.ndarray, np.float64]:
        """
        Return the gradient of the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : np.ndarray
            The weight array
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
            dw += (self._y_hat(X[i], w, b) - Y[i]) * X[i]
            db += self._y_hat(X[i], w, b) - Y[i]

        return dw / m, db / m
        """

        m = X.shape[0]
        dw = np.dot(X.T, self._y_hat(X, w, b) - Y) / m
        db = np.sum(self._y_hat(X, w, b) - Y) / m
        return dw, db

    def _printIteration(self, i: int) -> None:
        """
        Print the current iteration.

        Parameters
        ----------
        i : int
            The current iteration
        """
        n = len(str(self._num_iterations)) + 1
        print(f"Iteration: {i:{n}n} | Cost: {self._J_history[-1]:0.6e}")

    def fit(
        self, X: np.ndarray, Y: np.ndarray, w: np.ndarray = None, b: np.float64 = 0
    ) -> None:
        """
        Train the model given X and Y.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : np.ndarray, optional
            The weight array, by default None
        b : np.float64, optional
            The intercept, by default 0

        Raises
        ------
        ValueError
            If the number of rows in X and Y are not equal

        Notes
        -----
        If w is None, then it will be initialized to an array of zeros.

        If copy_X is True, then X will be copied before training.

        The cost function and parameters will be saved in J_history and p_history, respectively.

        If debug is True, then the cost function will be printed every 10% of the iterations.

        The cost function is computed using the mean squared error.

        The gradient is computed using the mean squared error.
        """

        n = X.shape[1] if len(X.shape) == 2 else 1

        if self._copy_X:
            X = copy.deepcopy(X)

        self._weights = np.zeros(n) if w is None else w
        self._intercept = b

        if self._X_scalar is not None:
            X = self._X_scalar.fit_transform(X)

        if self._Y_scalar is not None:
            Y = self._Y_scalar.fit_transform(Y).reshape(-1)

        self._J_history = [self._cost(X, Y, self._weights, self._intercept)]
        self._p_history = []

        for i in range(self._num_iterations):
            dw, db = self._gradient(X, Y, self._weights, self._intercept)
            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            self._J_history.append(self._cost(X, Y, self._weights, self._intercept))
            self._p_history.append((self._weights, self._intercept))

            if self._debug and i % (self._num_iterations // 10) == 0:
                self._printIteration(i)

        if self._debug:
            self._printIteration(self._num_iterations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted values given X.

        Parameters
        ----------
        X : np.ndarray
            The input array

        Returns
        -------
        np.ndarray
            The predicted values
        """

        if self._X_scalar is not None:
            X = self._X_scalar.fit_transform(X)

        predicton = self._y_hat(X, self._weights, self._intercept)

        if self._Y_scalar is not None:
            predicton = self._Y_scalar.inverse_transform(predicton)

        return predicton

    def get_cost_history(self) -> np.ndarray:
        """
        Return the history of the cost function.

        Returns
        -------
        np.ndarray
            The history of the cost function
        """
        return np.array(self._J_history)

    def get_parameter_history(self) -> np.ndarray:
        """
        Return the history of the parameters.

        Returns
        -------
        np.ndarray
            The history of the parameters
        """
        return np.array(self._p_history)

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

    def cost(
        self, X: np.ndarray, Y: np.ndarray, w: np.ndarray = None, b: np.float64 = None
    ) -> np.float64:
        """
        Return the cost for given X and Y.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : np.ndarray, optional
            The weight array, by default None
        b : np.float64, optional
            The intercept, by default None

        Returns
        -------
        np.float64
            The computed cost
        """

        if self._X_scalar is not None:
            X = self._X_scalar.fit_transform(X)

        if self._Y_scalar is not None:
            Y = self._Y_scalar.fit_transform(Y)

        if w is None:
            w = self._weights

        if b is None:
            b = self._intercept

        return self._cost(X, Y, w, b)
