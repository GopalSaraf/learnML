import copy
from typing import Tuple, Union, List
import numpy as np

from ..interfaces import IRegression, IFeatureEngineering


class LogisticRegression(IRegression):
    """
    # Logistic Regression Model

    Logistic Regression is a fundamental classification algorithm used to model the probability of a binary outcome. It's widely employed in machine learning for binary classification tasks and offers insights into the relationship between input features and class probabilities.

    ---

    ## Mathematical Approach

    Logistic Regression aims to predict the probability of a binary outcome by modeling it as a sigmoid function of a linear combination of input features. The equation takes the form:

    ```
    P(y=1 | X) = 1 / (1 + e^(-z))
    ```

    Where `P(y=1 | X)` is the probability of the positive class given input `X`, and `z` is the linear combination of input features, weights, and an intercept.

    ---

    ## Usage

    To use the Logistic Regression model, follow these steps:

    1. Import the `LogisticRegression` class from the appropriate module.
    2. Create an instance of the `LogisticRegression` class, specifying hyperparameters.
    3. Fit the model to your training data using the `fit` method.
    4. Make predictions on new data using the `predict` method.
    5. Evaluate the model's performance using the `score` method.

    ```python
    from learnML.classification import LogisticRegression

    # Create an instance of LogisticRegression
    model = LogisticRegression(learning_rate=0.001, n_iterations=1000)

    # Fit the model to training data
    model.fit(X_train, Y_train)

    # Make predictions on new data
    predictions = model.predict(X_test)

    # Calculate the model's score
    model_score = model.score(X_test, Y_test)
    ```

    ---

    ## Advantages

    - Simple and efficient
    - Can be updated easily with new data using stochastic gradient descent
    - Outputs have a nice probabilistic interpretation
    - Can be regularized to avoid overfitting
    - Works well with high dimensional data
    - Works well with sparse data

    ## Disadvantages

    - Not suitable for large number of features
    - Not suitable for non-linear problems

    ---
    """

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        n_iterations: int = 1000,
        lambda_: np.float64 = 0,
        x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None,
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

    def _sigmoid(self, z: np.float64) -> np.float64:
        """
        ### Sigmoid function

        Parameters
        ----------

        `z` : np.float64
        - The input

        Returns
        -------

        `np.float64`
        - The sigmoid of z

        ---
        """
        # 1 / (1 + e^(-z))
        return 1 / (1 + np.exp(-z))

    def _y_hat(
        self, X: np.ndarray, W: np.ndarray, b: np.float64
    ) -> Union[np.float64, np.ndarray]:
        """
        ### Return the predicted value of y given X, W, and b.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_features,) or (n_samples, n_features)

        `W` : np.ndarray
        - The weight array of shape (n_features,)

        `b` : np.float64
        - The intercept value


        Returns
        -------

        `np.float64`
        - The predicted value of y
        - If X is a 1D array, return a scalar
        - If X is a 2D array, return an array of shape (n_samples,)

        ---
        """
        z = np.dot(X, W) + b
        return self._sigmoid(z)

    def _cost(
        self, X: np.ndarray, y: np.float64, W: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        ### Return the cost of the model given X, y, W, and b.
        (Cross-entropy loss)

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `y` : np.float64
        - The output array of shape (n_samples,)

        `W` : np.ndarray
        - The weight array of shape (n_features,)

        `b` : np.float64
        - The intercept value


        Returns
        -------

        `np.float64`
        - The cost of the model

        ---
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
        ### Return the gradient of the model given X, y, W, and b.

        Parameters
        ----------

        `x` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `y` : np.float64
        - The output array of shape (n_samples,)

        `w` : np.ndarray
        - The weight array of shape (n_features,)

        `b` : np.float64
        - The intercept value


        Returns
        -------

        `Union[np.ndarray, np.float64]`
        - The gradient of the model with respect to w and b

        ---
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

    def _validate_data(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ### Return the input and output arrays of the model.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `Y` : np.ndarray, optional
        - The output array of shape (n_samples,)


        Returns
        -------

        `Tuple[np.ndarray, np.ndarray]`
        - The input and output arrays of the model
        - X has shape (n_samples, n_features)
        - Y has shape (n_samples,)

        ---
        """
        if self._copy_x:
            X = copy.deepcopy(X)

        X = self.__get_numpy_array(X)
        Y = self.__get_numpy_array(Y) if Y is not None else None

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for scalar in self._x_scalar:
            X = scalar.transform(X)

        if Y is not None:
            if Y.ndim == 2:
                Y = Y.reshape(-1)

        if Y is None:
            return X
        return X, Y

    def fit(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0.0
    ) -> None:
        """
        ### Fit the model to the data.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features) or (n_samples,)

        `Y` : np.ndarray
        - The output array of shape (n_samples,) or (n_samples, 1)

        `w` : np.ndarray, optional
        - The weight array, by default None
        - If None, then the weight array will be initialized to an array of
            zeros of shape (n_features,)
        - If not None, then the weight array will be initialized to the given
            array

        `b` : np.float64, optional
        - The intercept, by default 0.0
        - If None, then the intercept will be initialized to 0.0
        - If not None, then the intercept will be initialized to the given
            value


        Returns
        -------

        None

        ---
        """
        X, Y = self._validate_data(X, Y)

        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

        # Initialize weights and intercept
        self._weights = np.zeros(X.shape[1]) if W is None else W
        self._intercept = b

        self._cost_history = np.array(
            [self._cost(X, Y, self._weights, self._intercept)]
        )
        self._params_history = np.array(
            [[self._weights, self._intercept]], dtype=object
        )

        # Iterate and update weights and intercept
        for i in range(self._n_iterations):
            # Calculate gradient
            dw, db = self._gradient(X, Y, self._weights, self._intercept)

            # Update weights and intercept
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        ### Predict the probability of the output given the input.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features) or (n_features,)


        Returns
        -------

        `np.ndarray`
        - The predicted output array of shape (n_samples,) or (1,)

        ---
        """
        assert self._weights is not None and self._intercept is not None, (
            "The model must be trained before making predictions. "
            "Call the fit method first."
        )

        X = self._validate_data(X)
        # Return the probability of the output being 1
        return self._y_hat(X, self._weights, self._intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ### Predict the output given the input.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features) or (n_features,)


        Returns
        -------

        `np.ndarray`
        - The predicted output array of shape (n_samples,) or (1,)

        ---
        """
        # Return 1 if the probability of the output being 1 is greater than or equal to 0.5
        # Return 0 otherwise
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

    def score(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = None
    ) -> np.float64:
        """
        ### Return the cost of the model given X, Y, W, and b.

        Parameters
        ----------
        `X` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `Y` : np.ndarray
        - The output array of shape (n_samples,)

        `W` : np.ndarray, optional
        - The weight array of shape (n_features,), by default None
        - If None, then the weight array will be cosidered as the trained
            weight array

        `b` : np.float64, optional
        - The intercept, by default None
        - If None, then the intercept will be cosidered as the trained
            intercept


        Returns
        -------
        `np.float64`
        - The cost of the model

        ---
        """
        X, Y = self._validate_data(X, Y)

        W = self._weights if W is None else W
        b = self._intercept if b is None else b

        return self._cost(X, Y, W, b)
