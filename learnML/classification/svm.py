import numpy as np
from typing import Union, Tuple, List

from ..interfaces import IRegression, IFeatureEngineering


class LinearSVC(IRegression):
    """
    Support Vector Classifier

    Advantages
    ----------
    - Effective in high dimensional spaces
    - Works well with small number of samples
    - Works efficiently when there is a clear margin of separation between classes
    - Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient

    Disadvantages
    -------------
    - Not suitable for large number of samples (training time is higher)
    - Not suitable for noisy data with overlapping classes
    """

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        lambda_: np.float64 = 0.01,
        n_iterations: int = 1000,
        x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None,
        debug: bool = True,
        copy_x: bool = True,
    ):
        """
        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate of the model, by default 0.001
        lambda_param : np.float64, optional
            The regularization parameter of the model, by default 0.01
        n_iterations : int, optional
            The number of iterations to train the model, by default 1000
        x_scalar : Union[IFeatureEngineering, List[IFeatureEngineering]], optional
            The feature engineering for the input data, by default None
        debug : bool, optional
            Whether to print the training progress, by default True
        copy_x : bool, optional
            Whether to copy the training data, by default True
        """
        super().__init__(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            debug=debug,
            copy_x=copy_x,
        )
        self._lambda = lambda_

        if x_scalar is None:
            x_scalar = []
        elif isinstance(x_scalar, IFeatureEngineering):
            x_scalar = [x_scalar]
        self._x_scalar = x_scalar

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
        cost += self._lambda * np.dot(W.T, W)

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
                dw += 2 * self._lambda * W
            else:
                # Gradient of the hinge loss function
                # -y * x_i
                dw += 2 * self._lambda * W - np.dot(x_i, y[idx])
                db += y[idx]

        dw /= n_samples
        db /= n_samples

        return dw, db

    def _validate_data(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the X and y data

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)
        Y : np.ndarray, optional
            The target data of shape (n_samples, 1)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The X and Y data
        """
        if self._copy_x:
            X = np.copy(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for scalar in self._x_scalar:
            X = scalar.transform(X)

        if Y is not None:
            if Y.ndim == 2:
                Y = Y.reshape(-1)

            Y = np.where(Y <= 0, -1, 1)

        if Y is None:
            return X
        return X, Y

    def fit(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0.0
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
        assert X.shape[0] == Y.shape[0], "The number of samples must be equal"

        X, Y = self._validate_data(X, Y)

        self._weights = np.zeros(X.shape[1]) if W is None else W
        self._intercept = b

        self._cost_history = np.array(
            [self._cost(X, Y, self._weights, self._intercept)]
        )
        self._params_history = np.array(
            [[self._weights, self._intercept]], dtype=object
        )

        for i in range(self._n_iterations):
            dw, db = self._gradient(X, Y, self._weights, self._intercept)

            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

            # Save cost and params history
            cost = self._cost(X, Y, self._weights, self._intercept)

            if cost == np.nan or cost == np.inf:
                raise ValueError(
                    "Gradient descent failed. Try normalizing the input array or reducing the learning rate. "
                    "If the problem persists, try reducing the number of iterations."
                )

            self._cost_history = np.append(self._cost_history, cost)
            self._params_history = np.append(
                self._params_history,
                np.array([[self._weights, self._intercept]], dtype=object),
                axis=0,
            )

            if self._debug and i % self._debug_freq == 0:
                self._debug_print(i, cost)

        if self._debug:
            self._debug_print(self._n_iterations, cost)

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
        X = self._validate_data(X)

        # Return 1 if the prediction is greater than or equal to 0, otherwise return -1
        return np.where(self._y_hat(X, self._weights, self._intercept) >= 0, 1, 0)

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
        X, Y = self._validate_data(X, Y)

        W = self._weights if W is None else W
        b = self._intercept if b is None else b

        return self._cost(X, Y, W, b)
