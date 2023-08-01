from typing import Tuple, Union
import numpy as np

from ..interfaces import IModel, IFeatureEngineering


class UnivariateLinearRegression(IModel):
    """Univariate Linear Regression model."""

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        num_iterations: int = 10000,
        debug: bool = True,
        copy_X: bool = True,
        X_scalar: IFeatureEngineering = None,
        Y_scalar: IFeatureEngineering = None,
    ) -> None:
        """
        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate, by default 0.001
        num_iterations : int, optional
            The number of iterations, by default 10000
        lambda_ : np.float64, optional
            The regularization parameter, by default 0
        debug : bool, optional
            Whether to print debug information, by default True
        copy_X : bool, optional
            Whether to copy the input data, by default True
        X_scalar : IFeatureEngineering, optional
            The feature scaling object for the input data, by default None
        Y_scalar : IFeatureEngineering, optional
            The feature scaling object for the output data, by default None
        """
        self._learning_rate = learning_rate
        self._num_iterations = num_iterations
        self._debug = debug
        self._copy_X = copy_X
        self._X_scalar = X_scalar
        self._Y_scalar = Y_scalar

        self._weight: np.float64 = None
        self._intercept: np.float64 = None

        self._debug_freq = num_iterations // 10

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
        self, X: np.ndarray, Y: np.ndarray, w: np.float64, b: np.float64
    ) -> np.float64:
        """
        Return the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples,)
        Y : np.ndarray
            The output array of shape (n_samples,)
        w : np.float64
            The weight
        b : np.float64
            The intercept

        Returns
        -------
        np.float64
            The computed cost
        """

        m = X.shape[0]
        cost = 0
        for i in range(m):
            y_hat_i = self._y_hat(X[i], w, b)
            cost_i = (y_hat_i - Y[i]) ** 2
            cost += cost_i
        cost /= m
        return cost

    def _gradient(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.float64, np.float64]:
        """
        Return the gradient of the cost function given X and Y.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples,)
        Y : np.ndarray
            The output array of shape (n_samples,)

        Returns
        -------
        tuple
            The gradient of the cost function with respect to w and b
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

    def _getXandY(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the input and output arrays.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples,)
        Y : np.ndarray
            The output array of shape (n_samples,)

        Returns
        -------
        tuple
            The input and output arrays
        """

        if len(X.shape) == 2:
            X = X.reshape(-1)

        if len(Y.shape) == 2:
            Y = Y.reshape(-1)

        if self._copy_X:
            X = np.copy(X)

        if self._X_scalar is not None:
            X = self._X_scalar.transform(X)

        if self._Y_scalar is not None:
            Y = self._Y_scalar.transform(Y)

        return X, Y

    def fit(
        self, X: np.ndarray, Y: np.ndarray, w: np.float64 = 0.0, b: np.float64 = 0.0
    ) -> None:
        """
        Train the model given X and Y.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples,)
        Y : np.ndarray
            The output array of shape (n_samples,)
        w : np.float64, optional
            The initial weight, by default 0.0
        b : np.float64, optional
            The initial intercept, by default 0.0

        Returns
        -------
        None
        """

        assert len(X.shape) == 1 or (
            len(X.shape) == 2 and X.shape[1] == 1
        ), "X must be a 1D or 2D array with shape (n_samples,) or (n_samples, 1)"

        assert len(Y.shape) == 1 or (
            len(Y.shape) == 2 and Y.shape[1] == 1
        ), "Y must be a 1D or 2D array with shape (n_samples,) or (n_samples, 1)"

        X, Y = self._getXandY(X, Y)

        self._weight = w
        self._intercept = b
        self._J_history = [self._cost(X, Y, self._weight, self._intercept)]
        self._p_history = []

        for i in range(self._num_iterations):
            dw, db = self._gradient(X, Y)
            self._weight -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            self._J_history.append(self._cost(X, Y, self._weight, self._intercept))
            self._p_history.append((self._weight, self._intercept))

            if self._debug and i % self._debug_freq == 0:
                self._printIteration(i)

        if self._debug:
            self._printIteration(self._num_iterations)

    def predict(
        self, X: Union[np.ndarray, np.float64]
    ) -> Union[np.ndarray, np.float64]:
        """
        Return the predicted value of y given x.

        Parameters
        ----------
        X : Union[np.ndarray, np.float64]
            The input value or array of shape (n_samples,)

        Returns
        -------
        Union[np.ndarray, np.float64]
            The predicted value or array of shape (n_samples,)
        """
        assert self._weight is not None and self._intercept is not None, (
            "The model must be trained before making predictions. "
            "Call the fit method first."
        )

        isXScalar = isinstance(X, np.float64) or isinstance(X, int)

        if isinstance(X, np.ndarray):
            assert len(X.shape) == 1 or (
                len(X.shape) == 2 and X.shape[1] == 1
            ), "X must be a 1D or 2D array with shape (n_samples,) or (n_samples, 1)"
        else:
            X = np.array([X])

        if self._X_scalar is not None:
            X = self._X_scalar.transform(X)

        predictions = [self._y_hat(x, self._weight, self._intercept) for x in X]

        if self._Y_scalar is not None:
            predictions = self._Y_scalar.inverse_transform(predictions)

        if isXScalar:
            return predictions[0]
        else:
            return np.array(predictions)

    def score(
        self, X: np.ndarray, Y: np.ndarray, w: np.float64 = None, b: np.float64 = None
    ) -> np.float64:
        """
        Return the cost function given X, Y, w, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples,)
        Y : np.ndarray
            The output array of shape (n_samples,)
        w : np.float64, optional
            The weight, by default None
        b : np.float64, optional
            The intercept, by default None

        Returns
        -------
        np.float64
            The computed cost
        """
        X, Y = self._getXandY(X, Y)

        if w is None:
            w = self._weight

        if b is None:
            b = self._intercept

        return self._cost(X, Y, w, b)

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

    def _printIteration(self, iteration: int) -> None:
        """
        Print the current iteration and cost.

        Parameters
        ----------
        iteration : int
            The current iteration
        """
        n = len(str(self._num_iterations)) + 1
        cost = self._J_history[-1]
        print(f"Iteration: {iteration:{n}n} | Cost: {cost:0.6e}")
