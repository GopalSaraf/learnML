import numpy as np
from typing import Union, Tuple

from ..interfaces import IModel, IFeatureEngineering


class LinearSVC(IModel):
    """Support Vector Classifier"""

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        lambda_param: np.float64 = 0.01,
        num_iterations: int = 1000,
        debug: bool = True,
        copy_X: bool = True,
        X_scalar: IFeatureEngineering = None,
    ):
        """
        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate of the model, by default 0.001
        lambda_param : np.float64, optional
            The regularization parameter of the model, by default 0.01
        num_iterations : int, optional
            The number of iterations to train the model, by default 1000
        debug : bool, optional
            Whether to print the training progress, by default True
        copy_X : bool, optional
            Whether to copy the training data, by default True
        X_scalar : IFeatureEngineering, optional
            The feature engineering method to be applied to the training data, by default None
        """

        self._learning_rate = learning_rate
        self._lambda_param = lambda_param
        self._num_iterations = num_iterations
        self._debug = debug
        self._copy_X = copy_X
        self._X_scalar = X_scalar

        self._weights: np.ndarray = None
        self._intercept: np.float64 = None

        self._cost_history: np.ndarray = None
        self._params_history: np.ndarray = None

        self._debug_freq = num_iterations // 10

    def _y_hat(self, X: np.ndarray, W: np.ndarray, b: np.float64) -> np.ndarray:
        """
        Calculate the predicted value

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)
        W : np.ndarray
            The weights of shape (n_features,)
        b : np.float64
            The intercept

        Returns
        -------
        np.ndarray
            The predicted value of shape (n_samples, 1)
        """
        return np.dot(X, W) - b

    def _cost(
        self, X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        Calculate the cost of the model
        (Hinge loss)

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)
        y : np.ndarray
            The target data of shape (n_samples, 1)
        W : np.ndarray
            The weights of shape (n_features,)
        b : np.float64
            The intercept

        Returns
        -------
        np.float64
            The cost
        """
        n_samples = X.shape[0]
        y_hat = self._y_hat(X, W, b)

        # Hinge loss function
        # max(0, 1 - y * y_hat)
        loss = np.maximum(0, 1 - y * y_hat)

        # Calculate the cost
        # 1/n * sum(max(0, 1 - y * y_hat)) + lambda * ||W||^2
        cost = np.sum(loss) / n_samples
        cost += self._lambda_param * np.dot(W.T, W)

        return cost

    def _gradient(
        self, X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64
    ) -> Union[np.ndarray, np.float64]:
        """
        Calculate the gradient of the model

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)
        y : np.ndarray
            The target data of shape (n_samples, 1)
        W : np.ndarray
            The weights of shape (n_features,)
        b : np.float64
            The intercept

        Returns
        -------
        Union[np.ndarray, np.float64]
            The gradient of the model
        """
        n_samples = X.shape[0]
        y_hat = self._y_hat(X, W, b)

        # Calculate the gradient
        # 1/n * sum(max(0, 1 - y * y_hat)) + lambda * ||W||^2
        dw = np.zeros(W.shape)
        db = 0

        for idx, x_i in enumerate(X):
            if y[idx] * y_hat[idx] >= 1:
                # Gradient of the regularization term
                # 2 * lambda * W
                dw += 2 * self._lambda_param * W
            else:
                # Gradient of the hinge loss function
                # -y * x_i
                dw += 2 * self._lambda_param * W - np.dot(x_i, y[idx])
                db += y[idx]

        dw /= n_samples
        db /= n_samples

        return dw, db

    def _getXandY(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the X and y data

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)
        y : np.ndarray
            The target data of shape (n_samples, 1)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The X and y data
        """
        if self._copy_X:
            X = np.copy(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if len(y.shape) == 2:
            y = y.reshape(-1)

        if self._X_scalar:
            X = self._X_scalar.transform(X)

        y = np.where(y <= 0, -1, 1)

        return X, y

    def fit(
        self, X: np.ndarray, y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0.0
    ) -> None:
        """
        Fit the model to the data

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)
        y : np.ndarray
            The target data of shape (n_samples, 1) or (n_samples,)
        W : np.ndarray, optional
            The weights of shape (n_features,), by default None
        b : np.float64, optional
            The intercept, by default 0.0
        """
        assert X.shape[0] == y.shape[0], "The number of samples must be equal"

        X, y = self._getXandY(X, y)

        self._weights = np.zeros(X.shape[1]) if W is None else W
        self._intercept = b

        self._cost_history = [self._cost(X, y, self._weights, self._intercept)]
        self._params_history = []

        for i in range(self._num_iterations):
            dw, db = self._gradient(X, y, self._weights, self._intercept)

            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            self._cost_history.append(self._cost(X, y, self._weights, self._intercept))
            self._params_history.append((self._weights, self._intercept))

            if self._debug and i % self._debug_freq == 0:
                self._printIteration(i)

        if self._debug:
            self._printIteration(self._num_iterations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target data

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The predicted target data of shape (n_samples,)
        """
        X, _ = self._getXandY(X, np.zeros(X.shape[0]))

        # Return 1 if the prediction is greater than or equal to 0, otherwise return -1
        return np.where(
            self._y_hat(X, self._weights, self._intercept) >= 0, 1, -1
        ).reshape(-1)

    def get_cost_history(self) -> np.ndarray:
        """
        Get the cost history

        Returns
        -------
        np.ndarray
            The cost history
        """
        return np.array(self._cost_history)

    def get_params_history(self) -> np.ndarray:
        """
        Get the parameters history

        Returns
        -------
        np.ndarray
            The parameters history
        """
        return np.array(self._params_history)

    def get_weights(self) -> np.ndarray:
        """
        Get the weights

        Returns
        -------
        np.ndarray
            The weights
        """
        return self._weights

    def get_intercept(self) -> np.float64:
        """
        Get the intercept

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
