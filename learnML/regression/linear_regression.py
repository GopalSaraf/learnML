import copy
import numpy as np
from typing import Tuple, Union, List

from ..interfaces import IRegression, IFeatureEngineering


class LinearRegression(IRegression):
    """
    # Linear Regression Model

    Linear Regression is a fundamental supervised machine learning algorithm used to model the relationship between one or more independent variables and a continuous target variable. It serves as a cornerstone for predictive modeling and provides insights into how changes in input features affect the target variable. Linear Regression assumes that the target variable can be expressed as a linear combination of the input features, facilitating interpretation and prediction.

    ---

    ## Mathematical Approach

    Linear Regression aims to find the best-fitting linear equation that predicts the target variable. The equation takes the form:

    ```
    y = b + w1 * x1 + w2 * x2 + ... + wn * xn
    ```

    Where:

    - `y` is the predicted target variable.
    - `b` is the intercept (bias term).
    - `w1, w2, ..., wn` are the weights assigned to each input feature `x1, x2, ..., xn`.

    The goal of Linear Regression is to find the optimal values for `b` and the weights that minimize the difference between predicted and actual target values.

    ---

    ## Usage

    To use the Linear Regression model, follow these steps:

    1. Import the `LinearRegression` class from the appropriate module.
    2. Create an instance of the `LinearRegression` class, specifying hyperparameters.
    3. Fit the model to your training data using the `fit` method.
    4. Make predictions on new data using the `predict` method.
    5. Evaluate the model's performance using the `score` method.

    ```python
    from learnML.regression import LinearRegression

    # Create an instance of LinearRegression
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)

    # Fit the model to training data
    model.fit(X_train, Y_train)

    # Make predictions on new data
    predictions = model.predict(X_test)

    # Calculate the model's score
    model_score = model.score(X_test, Y_test)
    ```

    ---

    ## Advantages

    - Easy to implement
    - Easy to interpret the output
    - Computationally cheap

    ## Disadvantages

    - Poor performance on non-linear data
    - Sensitive to outliers
    - Sensitive to overfitting

    ---
    """

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        n_iterations: int = 1000,
        lambda_: np.float64 = 0,
        x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None,
        y_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None,
        debug: bool = True,
        copy_x: bool = True,
    ) -> None:
        """
        Parameters
        ----------

        `learning_rate` : np.float64, optional
        - The learning rate, by default 0.001
        - The learning rate determines how much the weights are updated at each iteration
        - A low learning rate will take longer to converge, but a high learning rate may overshoot the optimal solution

        `n_iterations` : int, optional
        - The number of iterations, by default 1000
        - The number of iterations determines how many times the weights are updated
        - A higher number of iterations will take longer to converge, but a lower number of iterations may not be enough to converge

        `lambda_` : np.float64, optional
        - The regularization parameter, by default 0
        - The regularization parameter helps prevent overfitting by penalizing large weights
        - A higher regularization parameter will penalize large weights more, but a lower regularization parameter may not be enough to prevent overfitting

        `x_scalar` : Union[IFeatureEngineering, List[IFeatureEngineering]], optional
        - The feature engineering for the input data, by default None
        - If a list is provided, the feature engineering will be applied in the order provided
        - If a single feature engineering is provided, it will be applied to all input data

        `y_scalar` : Union[IFeatureEngineering, List[IFeatureEngineering]], optional
        - The feature engineering for the output data, by default None
        - If a list is provided, the feature engineering will be applied in the order provided
        - If a single feature engineering is provided, it will be applied to all output data

        `debug` : bool, optional
        - Whether to print debug messages, by default True
        - Debug messages include the cost at each iteration

        `copy_x` : bool, optional
        - Whether to copy the input array, by default True
        - If False, the input array will be overwritten

        ---
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

        if y_scalar is None:
            y_scalar = []
        elif isinstance(y_scalar, IFeatureEngineering):
            y_scalar = [y_scalar]
        self._y_scalar = y_scalar

    def _y_hat(self, X: np.ndarray, W: np.ndarray, b: np.float64) -> np.float64:
        """
        ### Return the predicted value of y given x, w, and b.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_features,)

        `W` : np.ndarray
        - The weight array of shape (n_features,)

        `b` : np.float64
        - The intercept


        Returns
        -------

        `np.float64`
        - The predicted value of y

        ---
        """
        return np.dot(X, W) + b

    def _cost(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        ### Return the cost function given X, Y, w, and b.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `Y` : np.ndarray
        - The output array of shape (n_samples,)

        `W` : np.ndarray
        - The weight array of shape (n_features,)

        `b` : np.float64
        - The intercept


        Returns
        -------

        `np.float64`
        - The computed cost

        ---
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
        ### Return the gradient of the cost function given X, Y, w, and b.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `Y` : np.ndarray
        - The output array of shape (n_samples,)

        `W` : np.ndarray
        - The weight array of shape (n_features,)

        `b` : np.float64
        - The intercept


        Returns
        -------

        `Tuple[np.ndarray, np.float64]`
        - The computed gradient

        ---
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

    def _validate_data(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ### Return the input and output arrays.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `Y` : np.ndarray, optional
        - The output array of shape (n_samples,) or (n_samples, 1)


        Returns
        -------

        `Tuple[np.ndarray, np.ndarray]`
        - The input and output arrays

        ---
        """
        if self._copy_x:
            X = copy.deepcopy(X)

        X = self.__get_numpy_array(X)
        Y = self.__get_numpy_array(Y) if Y is not None else None

        if Y is not None:
            assert (
                X.shape[0] == Y.shape[0]
            ), "X and Y must have the same number of samples"

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for scalar in self._x_scalar:
            X = scalar.transform(X)

        if Y is not None:
            if Y.ndim == 2:
                Y = Y.reshape(-1)

            for scalar in self._y_scalar:
                Y = scalar.transform(Y)

        if Y is None:
            return X
        return X, Y

    def fit(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0
    ) -> None:
        """
        ### Train the model given X and Y.


        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features) or (n_samples,)
        - If the shape is (n_samples,), then the model will be trained as a
            univariate linear regression model
        - If the shape is (n_samples, n_features), then the model will be
            trained as a multivariate linear regression model

        `Y` : np.ndarray
        - The output array of shape (n_samples,) or (n_samples, 1)

        `w` : np.ndarray, optional
        - The weight array, by default None
        - If None, then the weight array will be initialized to an array of
            zeros of shape (n_features,)
        - If not None, then the weight array will be initialized to the given
            array

        `b` : np.float64, optional
        - The intercept, by default 0
        - If None, then the intercept will be initialized to 0
        - If not None, then the intercept will be initialized to the given
            value


        Returns
        -------

        None

        ---
        """
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

            self._weights = self._weights.astype("float64")
            self._weights -= self._learning_rate * dw
            self._intercept -= self._learning_rate * db

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
        ### Return the predicted values given X.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features) or (n_samples,)


        Returns
        -------

        `np.ndarray`
        - The predicted values of shape (n_samples, 1) or (n_samples,)
        - If the model was trained as a univariate linear regression model,
            then the shape will be (n_samples,)
        - If the model was trained as a multivariate linear regression model,
            then the shape will be (n_samples, 1)

        ---
        """
        assert self._weights is not None and self._intercept is not None, (
            "The model must be trained before making predictions. "
            "Call the fit method first."
        )

        X = self._validate_data(X)

        predictions = [
            self._y_hat(X[i], self._weights, self._intercept) for i in range(X.shape[0])
        ]

        for scalar in self._y_scalar:
            predictions = scalar.inverse_transform(predictions)

        return np.array(predictions)

    def score(
        self, X: np.ndarray, Y: np.ndarray, w: np.ndarray = None, b: np.float64 = None
    ) -> np.float64:
        """
        ### Return the cost for given X and Y.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features) or (n_samples,)

        `Y` : np.ndarray
        - The output array of shape (n_samples,) or (n_samples, 1)

        `w` : np.ndarray, optional
        - The weight array, by default None
        - If None, then the weight array will be cosidered as the trained
            weight array

        `b` : np.float64, optional
        - The intercept, by default None
        - If None, then the intercept will be cosidered as the trained
            intercept


        Returns
        -------

        `np.float64`
        - The computed cost

        ---
        """
        X, Y = self._validate_data(X, Y)

        w = self._weights if w is None else w
        b = self._intercept if b is None else b

        return self._cost(X, Y, w, b)
