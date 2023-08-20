import numpy as np
from typing import Tuple, Union, List

from ..interfaces import IRegression, IFeatureEngineering


class UnivariateLinearRegression(IRegression):
    """
    Univariate Linear Regression model.

    Advatanges
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
        learning_rate: np.float64 = 0.001,
        n_iterations: int = 1000,
        x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None,
        y_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None,
        debug: bool = True,
        copy_x: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate, by default 0.001
        n_iterations : int, optional
            The number of iterations, by default 1000
        x_scalar : Union[IFeatureEngineering, List[IFeatureEngineering]], optional
            The feature engineering for the input data, by default None
        y_scalar : Union[IFeatureEngineering, List[IFeatureEngineering]], optional
            The feature engineering for the output data, by default None
        debug : bool, optional
            Whether to print debug information, by default True
        copy_x : bool, optional
            Whether to copy the input data, by default True
        """
        super().__init__(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            debug=debug,
            copy_x=copy_x,
        )

        if x_scalar is None:
            x_scalar = []
        elif isinstance(x_scalar, IFeatureEngineering):
            x_scalar = [x_scalar]
        self._x_scalar = x_scalar

        if y_scalar is None:
            y_scalar = []
        elif isinstance(y_scalar, IFeatureEngineering):
            y_scalar = [y_scalar]
        self._y_scalar = y_scalar

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
        (Mean Squared Error)

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
        # Number of samples
        m = X.shape[0]
        cost = 0

        # cost = 1 / 2m * sum((y_hat_i - y_i) ^ 2)
        for i in range(m):
            y_hat_i = self._y_hat(X[i], w, b)
            cost_i = (y_hat_i - Y[i]) ** 2
            cost += cost_i
        cost /= 2 * m
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
        Tuple[np.float64, np.float64]
            The gradient of the cost function with respect to w and b
        """
        # Number of samples
        m = X.shape[0]
        dw = 0
        db = 0

        # dw = 1 / m * sum((y_hat_i - y_i) * x_i)
        # db = 1 / m * sum(y_hat_i - y_i)

        for i in range(m):
            y_hat_i = self._y_hat(X[i], self._weights, self._intercept)
            dw_i = (y_hat_i - Y[i]) * X[i]
            dw += dw_i
            db_i = y_hat_i - Y[i]
            db += db_i

        return dw / m, db / m

    def _validate_data(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the input and output arrays.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples,)
        Y : np.ndarray or None, optional
            The output array of shape (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The input and output arrays
        """
        # Check the shape of X and Y
        assert X.ndim == 1 or (
            X.ndim == 2 and X.shape[1] == 1
        ), "X must be a 1D or 2D array with shape (n_samples,) or (n_samples, 1)"

        # Copy the arrays if necessary
        if self._copy_x:
            X = np.copy(X)

        # Reshape the arrays if necessary
        if X.ndim == 2:
            X = X.reshape(-1)

        # Scale input and output if necessary
        for scalar in self._x_scalar:
            X = scalar.transform(X)

        if Y is not None:
            assert Y.ndim == 1 or (
                Y.ndim == 2 and Y.shape[1] == 1
            ), "Y must be a 1D or 2D array with shape (n_samples,) or (n_samples, 1)"

            if Y.ndim == 2:
                Y = Y.reshape(-1)

            for scalar in self._y_scalar:
                Y = scalar.transform(Y)

        if Y is None:
            return X
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
        X, Y = self._validate_data(X, Y)

        self._weights = w
        self._intercept = b

        self._cost_history = np.array(
            [self._cost(X, Y, self._weights, self._intercept)]
        )
        self._params_history = np.array([np.array([self._weights, self._intercept])])

        # Gradient descent
        for i in range(self._n_iterations):
            # Compute the gradient
            dw, db = self._gradient(X, Y)

            # Update the weight and intercept
            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            cost = self._cost(X, Y, self._weights, self._intercept)

            if cost == np.nan or cost == np.inf:
                raise ValueError(
                    "Gradient descent failed. Try normalizing the input array or reducing the learning rate. "
                    "If the problem persists, try reducing the number of iterations."
                )

            # Save the cost and parameters
            self._cost_history = np.append(self._cost_history, cost)
            self._params_history = np.append(
                self._params_history,
                np.array([[self._weights, self._intercept]]),
                axis=0,
            )

            # Print the cost and parameters
            if self._debug and i % self._debug_freq == 0:
                self._debug_print(i, cost)

        if self._debug:
            self._debug_print(self._n_iterations, cost)

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
        # Check if the model is trained
        assert self._weights is not None and self._intercept is not None, (
            "The model must be trained before making predictions. "
            "Call the fit method first."
        )

        isXScalar = isinstance(X, np.float64) or isinstance(X, int)

        if isinstance(X, np.ndarray):
            assert X.ndim == 1 or (
                X.ndim == 2 and X.shape[1] == 1
            ), "X must be a 1D or 2D array with shape (n_samples,) or (n_samples, 1)"
        else:
            X = np.array([X])

        for scalar in self._x_scalar:
            X = scalar.transform(X)

        predictions = [self._y_hat(x, self._weights, self._intercept) for x in X]

        for scalar in self._y_scalar:
            predictions = scalar.inverse_transform(predictions)

        return predictions[0] if isXScalar else np.array(predictions)

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
        X, Y = self._validate_data(X, Y)

        w = self._weights if w is None else w
        b = self._intercept if b is None else b

        return self._cost(X, Y, w, b)
