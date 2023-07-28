from typing import Tuple, Union
import numpy as np
import copy

from ..interfaces import IModel, IFeatureScaling


class LogisticRegression(IModel):
    """Logistic Regression Model"""

    def __init__(
        self,
        learning_rate: np.float64 = 0.0001,
        num_iterations: int = 10000,
        debug: bool = True,
        copy_X: bool = True,
        X_scalar: IFeatureScaling = None,
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
        super().__init__(learning_rate, num_iterations, X_scalar, None, debug, copy_X)

        self._weights: np.ndarray
        self._intercept: np.float64

    def _sigmoid(self, z: np.float64) -> np.float64:
        """
        Sigmoid function

        Parameters
        ----------
        z : np.float64
            The input

        Returns
        -------
        np.float64
            The sigmoid of z
        """
        return 1 / (1 + np.exp(-z))

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
        z = np.dot(x, w) + b
        return self._sigmoid(z)

    def _cost(
        self, x: np.ndarray, y: np.float64, w: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        Return the cost of the model given x, y, w, and b.

        Parameters
        ----------
        x : np.ndarray
            The input array
        y : np.float64
            The output
        w : np.ndarray
            The weight array
        b : np.float64
            The intercept

        Returns
        -------
        np.float64
            The cost of the model
        """
        """
        ALTERNATIVE IMPLEMENTATION
        --------------------------
        cost = 0.0
        m = x.shape[0]

        for i in range(m):
            y_hat = self._y_hat(x[i], w, b)
            pos_cost = np.dot(y[i], np.log(y_hat))
            neg_cost = np.dot(1 - y[i], np.log(1 - y_hat))
            cost += pos_cost + neg_cost

        return -cost / m
        """
        y_hat = self._y_hat(x, w, b)
        pos_cost = np.dot(y, np.log(y_hat))
        neg_cost = np.dot(1 - y, np.log(1 - y_hat))
        return -(pos_cost + neg_cost) / x.shape[0]

    def _gradient(
        self, x: np.ndarray, y: np.float64, w: np.ndarray, b: np.float64
    ) -> Union[np.ndarray, np.float64]:
        """
        Return the gradient of the model given x, y, w, and b.

        Parameters
        ----------
        x : np.ndarray
            The input array
        y : np.float64
            The output
        w : np.ndarray
            The weight array
        b : np.float64
            The intercept

        Returns
        -------
        Union[np.ndarray, np.float64]
            The gradient of the model
        """
        """
        ALTERNATIVE IMPLEMENTATION
        --------------------------
        dw = np.zeros(w.shape)
        db = 0.0
        m = x.shape[0]

        for i in range(m):
            y_hat = self._y_hat(x[i], w, b)
            dw += np.dot(x[i].T, y_hat - y[i])
            db += y_hat - y[i]

        return dw / m, db / m
        """
        y_hat = self._y_hat(x, w, b)
        dw = np.dot(x.T, y_hat - y)
        db = np.sum(y_hat - y)
        return dw / x.shape[0], db / x.shape[0]

    def _printIteration(self, iteration: int) -> None:
        """
        Print the current iteration and cost.

        Parameters
        ----------
        iteration : int
            The current iteration
        """
        n = len(str(self._num_iterations)) + 1
        print(f"Iteration: {iteration:{n}n} | Cost: {self._J_history[-1]:0.6e}")

    def fit(
        self, X: np.ndarray, Y: np.ndarray, w: np.float64 = 0.0, b: np.float64 = 0.0
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : np.float64, optional
            The initial weight, by default 0.0
        b : np.float64, optional
            The initial intercept, by default 0.0
        """
        n = X.shape[1]
        self._weights = np.full((n, 1), w)
        self._intercept = b

        if self._copy_X:
            X = copy.deepcopy(X)

        if self._X_scalar is not None:
            X = self._X_scalar.fit_transform(X)

        for i in range(self._num_iterations):
            dw, db = self._gradient(X, Y, self._weights, self._intercept)
            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db
            self._J_history.append(self._cost(X, Y, self._weights, self._intercept))
            self._p_history.append((self._weights, self._intercept))

            if self._debug and i % self._debug_freq == 0:
                self._printIteration(i)

        if self._debug:
            self._printIteration(self._num_iterations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output given the input.

        Parameters
        ----------
        X : np.ndarray
            The input array

        Returns
        -------
        np.ndarray
            The predicted output array
        """
        if self._copy_X:
            X = copy.deepcopy(X)

        if self._X_scalar is not None:
            X = self._X_scalar.transform(X)

        return np.where(self._y_hat(X, self._weights, self._intercept) >= 0.5, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of the output given the input.

        Parameters
        ----------
        X : np.ndarray
            The input array

        Returns
        -------
        np.ndarray
            The predicted output array
        """
        if self._copy_X:
            X = copy.deepcopy(X)

        if self._X_scalar is not None:
            X = self._X_scalar.transform(X)

        return self._y_hat(X, self._weights, self._intercept)

    def get_cost_history(self) -> np.ndarray:
        """
        Return the cost history.

        Returns
        -------
        np.ndarray
            The cost history
        """
        return np.array(self._J_history)

    def get_parameter_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the parameter history.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The parameter history
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
        Return the cost of the model given x, y, w, and b.

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
            The cost of the model
        """
        if w is None:
            w = self._weights
        if b is None:
            b = self._intercept

        if self._X_scalar is not None:
            X = self._X_scalar.transform(X)

        return self._cost(X, Y, w, b)
