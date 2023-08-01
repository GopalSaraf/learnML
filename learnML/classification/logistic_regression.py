from typing import Tuple, Union
import numpy as np

from ..interfaces import IModel, IFeatureEngineering


class LogisticRegression(IModel):
    """Logistic Regression Model"""

    def __init__(
        self,
        learning_rate: np.float64 = 0.0001,
        num_iterations: int = 10000,
        debug: bool = True,
        copy_X: bool = True,
        X_scalar: IFeatureEngineering = None,
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
        X_scalar : IFeatureEngineering, optional
            The feature scaling object for the input array, by default None
        """
        self._learning_rate = learning_rate
        self._num_iterations = num_iterations
        self._debug = debug
        self._copy_X = copy_X
        self._X_scalar = X_scalar

        self._weights: np.ndarray = None
        self._intercept: np.float64 = None

        self._cost_history: np.ndarray = None
        self._params_history: np.ndarray = None

        self._debug_freq = num_iterations // 10

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
        # 1 / (1 + e^(-z))
        return 1 / (1 + np.exp(-z))

    def _y_hat(
        self, X: np.ndarray, W: np.ndarray, b: np.float64
    ) -> Union[np.float64, np.ndarray]:
        """
        Return the predicted value of y given X, W, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_features,) or (n_samples, n_features)
        W : np.ndarray
            The weight array of shape (n_features,)
        b : np.float64
            The intercept value

        Returns
        -------
        np.float64
            The predicted value of y
            If X is a 1D array, return a scalar
            If X is a 2D array, return an array of shape (n_samples,)
        """
        z = np.dot(X, W) + b
        return self._sigmoid(z)

    def _cost(
        self, X: np.ndarray, y: np.float64, W: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        Return the cost of the model given X, y, W, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        y : np.float64
            The output array of shape (n_samples,)
        W : np.ndarray
            The weight array of shape (n_features,)
        b : np.float64
            The intercept value

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
        y_hat = self._y_hat(X, W, b)

        # Cost of logistic regression : -[ylog(y_hat) + (1-y)log(1-y_hat)] / m
        pos_cost = np.dot(y, np.log(y_hat))
        neg_cost = np.dot(1 - y, np.log(1 - y_hat))

        return -np.sum(pos_cost + neg_cost) / X.shape[0]

    def _gradient(
        self, X: np.ndarray, Y: np.float64, W: np.ndarray, b: np.float64
    ) -> Union[np.ndarray, np.float64]:
        """
        Return the gradient of the model given X, y, W, and b.

        Parameters
        ----------
        x : np.ndarray
            The input array of shape (n_samples, n_features)
        y : np.float64
            The output array of shape (n_samples,)
        w : np.ndarray
            The weight array of shape (n_features,)
        b : np.float64
            The intercept value

        Returns
        -------
        Union[np.ndarray, np.float64]
            The gradient of the model with respect to w and b
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
        y_hat = self._y_hat(X, W, b)

        # Gradient of logistic regression

        # dJ/dW = (1/m) * sum((y_hat - y) * x)
        # We use X.T to get the transpose of X so that we can multiply it with (y_hat - y)
        # transpose of X has shape (n_features, n_samples)
        # (y_hat - y) has shape (n_samples,)
        # The result of the multiplication has shape (n_features,)
        dJ_dW = np.dot(X.T, y_hat - Y) / X.shape[0]

        # dJ/db = (1/m) * sum(y_hat - y)
        # (y_hat - y) has shape (n_samples,)
        # The result of the multiplication has shape (1,)
        dJ_db = np.sum(y_hat - Y) / X.shape[0]

        return dJ_dW, dJ_db

    def _getXAndY(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the input and output arrays of the model.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        Y : np.ndarray
            The output array of shape (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The input and output arrays of the model
            X has shape (n_samples, n_features)
            Y has shape (n_samples,)
        """
        if self._copy_X:
            X = np.copy(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if len(Y.shape) == 2:
            Y = Y.reshape(-1)

        if self._X_scalar is not None:
            X = self._X_scalar.fit_transform(X)

        return X, Y

    def fit(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0.0
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        Y : np.ndarray
            The output array of shape (n_samples,)
        W : np.ndarray, optional
            The weight array of shape (n_features,), by default None
        b : np.float64, optional
            The intercept value, by default 0.0

        Returns
        -------
        None
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

        X, Y = self._getXAndY(X, Y)

        # Initialize weights and intercept
        self._weights = np.zeros(X.shape[1]) if W is None else W
        self._intercept = b

        self._cost_history = [self._cost(X, Y, self._weights, self._intercept)]
        self._params_history = []

        # Iterate and update weights and intercept
        for i in range(self._num_iterations):
            # Calculate gradient
            dw, db = self._gradient(X, Y, self._weights, self._intercept)

            # Update weights and intercept
            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            # Save cost and params history
            self._cost_history.append(self._cost(X, Y, self._weights, self._intercept))
            self._params_history.append((self._weights, self._intercept))

            if self._debug and i % self._debug_freq == 0:
                self._printIteration(i)

        if self._debug:
            self._printIteration(self._num_iterations)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of the output given the input.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        np.ndarray
            The predicted output array of shape (n_samples,) or (1,)
        """
        assert self._weights is not None and self._intercept is not None, (
            "The model must be trained before making predictions. "
            "Call the fit method first."
        )

        X, _ = self._getXAndY(X, np.zeros(X.shape[0]))
        # Return the probability of the output being 1
        return self._y_hat(X, self._weights, self._intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output given the input.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        np.ndarray
            The predicted output array of shape (n_samples,) or (1,)
        """
        # Return 1 if the probability of the output being 1 is greater than or equal to 0.5
        # Return 0 otherwise
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

    def get_cost_history(self) -> np.ndarray:
        """
        Return the cost history.

        Returns
        -------
        np.ndarray
            The cost history array
        """
        return np.array(self._cost_history)

    def get_parameter_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the parameter history.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The parameter history
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
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = None
    ) -> np.float64:
        """
        Return the cost of the model given X, Y, W, and b.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        Y : np.ndarray
            The output array of shape (n_samples,)
        W : np.ndarray, optional
            The weight array of shape (n_features,), by default None
        b : np.float64, optional
            The intercept, by default None

        Returns
        -------
        np.float64
            The cost of the model
        """
        X, Y = self._getXAndY(X, Y)

        W = self._weights if W is None else W
        b = self._intercept if b is None else b

        return self._cost(X, Y, W, b)

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
