from typing import Tuple
import numpy as np
import copy

from ..interfaces import IModel, IFeatureScaling


class UnivariateLinearRegression(IModel):
    """Univariate Linear Regression model."""

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        num_iterations: int = 10000,
        debug: bool = True,
        copy_X: bool = True,
        X_scalar: IFeatureScaling = None,
        Y_scalar: IFeatureScaling = None,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate of the model, by default 0.001
        num_iterations : int, optional
            The number of iterations to train the model, by default 10000
        debug : bool, optional
            Whether to print the iteration number, cost, and parameters, by default True
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

        self._weight: np.float64 = None
        self._intercept: np.float64 = None

    def _y_hat(self, x: np.float64, w: np.float64, b: np.float64) -> np.float64:
        """
        Return the predicted value given x, w, and b.

        Parameters
        ----------
        x : np.float64
            The input value
        w : np.float64
            The weight
        b : np.float64
            The intercept

        Returns
        -------
        np.float64
            The predicted value
        """
        return w * x + b

    def _cost(
        self, X: np.ndarray, Y: np.ndarray, w: np.float64 = None, b: np.float64 = None
    ) -> np.float64:
        """
        Return the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : np.float64, optional
            The weight, by default None
        b : np.float64, optional
            The intercept, by default None

        Returns
        -------
        np.float64
            The computed cost
        """

        m = X.shape[0]
        cost = 0

        if w is None:
            w = self._weight
        if b is None:
            b = self._intercept

        for i in range(m):
            y_hat_i = self._y_hat(X[i], w, b)
            cost_i = (y_hat_i - Y[i]) ** 2
            cost += cost_i
        return cost / (2 * m)

    def _gradient(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.float64, np.float64]:
        """
        Return the gradient of the cost function given X and Y.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array

        Returns
        -------
        tuple
            The gradient of the cost function
        """

        m = X.shape[0]
        dw = 0
        db = 0
        for i in range(m):
            y_hat_i = self._y_hat(X[i], self._weight, self._intercept)
            dw_i = (y_hat_i - Y[i]) * X[i]
            dw += dw_i
            db_i = y_hat_i - Y[i]
            db += db_i
        return dw / m, db / m

    def _printIteration(self, i: int) -> None:
        """
        Print the iteration number, cost, and parameters.

        Parameters
        ----------
        i : int
            The iteration number

        Returns
        -------
        None
        """
        n = len(str(self._num_iterations))
        print(f"Iteration: {i:{n}n} | Cost: {self._J_history[-1]:0.6e}")

    def fit(
        self, X: np.ndarray, Y: np.ndarray, w: np.float64 = 0.0, b: np.float64 = 0.0
    ) -> None:
        """
        Train the model given X and Y.

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

        Returns
        -------
        None

        Notes
        -----
        The model will be trained for the number of iterations specified in the constructor.

        The model will be trained using the learning rate specified in the constructor.

        The model will be trained using the gradient descent algorithm.

        """

        self._weight = w
        self._intercept = b
        self._J_history = [self._cost(X, Y)]
        self._p_history = []

        if self._copy_X:
            X = copy.deepcopy(X)

        if self._X_scalar is not None:
            X = self._X_scalar.fit_transform(X)

        if self._Y_scalar is not None:
            Y = self._Y_scalar.fit_transform(Y).reshape(-1)

        for i in range(self._num_iterations):
            dw, db = self._gradient(X, Y)
            self._weight -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            self._J_history.append(self._cost(X, Y))
            self._p_history.append((self._weight, self._intercept))

            if self._debug and i % (self._num_iterations // 10) == 0:
                self._printIteration(i)

        if self._debug:
            self._printIteration(self._num_iterations)

    def predict(self, x: np.float64) -> np.float64:
        """
        Return the predicted value of y given x.

        Parameters
        ----------
        x : np.float64
            The input value

        Returns
        -------
        np.float64
            The predicted value of y
        """
        if self._X_scalar is not None:
            x = self._X_scalar.transform(x)

        prediction = self._y_hat(x, self._weight, self._intercept)

        if self._Y_scalar is not None:
            prediction = self._Y_scalar.inverse_transform(prediction)

        return prediction

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted values of y given X.

        Parameters
        ----------
        X : np.ndarray
            The input array

        Returns
        -------
        np.ndarray
            The predicted values of y
        """
        return np.array([self.predict(x) for x in X])

    def cost(
        self, X: np.ndarray, Y: np.ndarray, w: np.float64 = None, b: np.float64 = None
    ) -> np.float64:
        """
        Return the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : np.float64, optional
            The weight, by default None
        b : np.float64, optional
            The intercept, by default None

        Returns
        -------
        np.float64
            The computed cost
        """

        if w is None:
            w = self._weight
        if b is None:
            b = self._intercept

        if self._X_scalar is not None:
            X = self._X_scalar.fit_transform(X)

        if self._Y_scalar is not None:
            Y = self._Y_scalar.fit_transform(Y)

        return self._cost(X, Y, w, b)

    def get_J_history(self) -> np.ndarray:
        """
        Return the history of the cost function.

        Returns
        -------
        np.ndarray
            The history of the cost function
        """
        return np.array(self._J_history)

    def get_p_history(self) -> np.ndarray:
        """
        Return the history of the parameters.

        Returns
        -------
        np.ndarray
            The history of the parameters
        """
        return np.array(self._p_history)

    def get_weight(self) -> np.float64:
        """
        Return the value of w.

        Returns
        -------
        np.float64
            The value of w
        """
        return self._weight

    def get_intercept(self) -> np.float64:
        """
        Return the value of b.

        Returns
        -------
        np.float64
            The value of b
        """
        return self._intercept
