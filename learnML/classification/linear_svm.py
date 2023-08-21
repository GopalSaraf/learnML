import numpy as np
from typing import Union, Tuple, List

from ..interfaces import IRegression, IFeatureEngineering


class LinearSVC(IRegression):
    """
    # Support Vector Classifier

    `LinearSVC` (Support Vector Classifier) is a classification algorithm that aims to separate data into two classes by finding a hyperplane that maximizes the margin between them. It's effective for high-dimensional data and can handle small sample sizes efficiently. This algorithm seeks to create a decision boundary by minimizing classification errors and maximizing the margin between the classes.

    ---

    ## Mathematical Approach

    `LinearSVC` aims to find a hyperplane that best separates two classes by minimizing a loss function known as the hinge loss. The decision boundary is represented as a linear combination of input features, weights, and an intercept:

    ```
    z = X * W - b
    ```

    Where:

    - `X` is the input data matrix of shape `(n_samples, n_features)`.
    - `W` is the weight vector of shape `(n_features,)`.
    - `b` is the intercept.

    The predicted class `y_hat` is determined based on the sign of `z`:

    ```
    y_hat = sign(z)
    ```

    ---

    ### Hinge Loss

    The hinge loss function measures the degree of violation of a sample's classification. For a sample `(x_i, y_i)`, where `x_i` is the input data and `y_i` is the true class label (-1 or 1), the hinge loss is defined as:

    ```
    loss_i = max(0, 1 - y_i * z_i)
    ```

    Where `z_i` is the linear combination for the `i`-th sample. The overall hinge loss for the entire dataset is the sum of individual hinge losses:

    ```
    loss = sum(max(0, 1 - y_i * z_i)) for all samples i
    ```

    ---

    ### Margin and Support Vectors

    The margin is the distance between the decision boundary and the closest data points. The goal is to maximize this margin while minimizing the hinge loss. Support vectors are the data points that are closest to the decision boundary and play a crucial role in defining the hyperplane.

    Maximizing the margin is equivalent to minimizing the norm of the weight vector `W`:

    ```
    min (||W|| / 2)
    ```

    ---

    ### Regularization

    Regularization is used to prevent overfitting by penalizing large weights. The regularization term is added to the loss function and is defined as:

    ```
    lambda_ * ||W||^2
    ```

    Where `lambda_` is the regularization parameter.

    ---

    ### Optimization

    The goal is to minimize the hinge loss and the regularization term. This is achieved by using gradient descent to iteratively update the weights and intercept. The gradient of the hinge loss function with respect to the weights `W` and intercept `b` is calculated for each sample. For correctly classified samples (`y * z >= 1`), only the regularization term contributes to the gradient. For misclassified samples (`y * z < 1`), both the regularization term and the hinge loss gradient contribute.

    The gradients are averaged over all samples and used to update the weights and intercept using the learning rate. This process is repeated for the specified number of iterations.

    ---

    ## Usage

    To use the `LinearSVC` model, follow these steps:

    1. Import the `LinearSVC` class from the appropriate module.
    2. Create an instance of the `LinearSVC` class, specifying hyperparameters.
    3. Fit the model to your training data using the `fit` method.
    4. Make predictions on new data using the `predict` method.
    5. Evaluate the model's performance using the `score` method.

    ```python
    from learnML.classification import LinearSVC

    # Create an instance of LinearSVC
    model = LinearSVC(learning_rate=0.001, lambda_=0.01, n_iterations=1000)

    # Fit the model to training data
    model.fit(X_train, Y_train)

    # Make predictions on new data
    predictions = model.predict(X_test)

    # Calculate the model's score
    model_score = model.score(X_test, Y_test)
    ```

    ---

    ## Advantages

    - Effective in high dimensional spaces
    - Works well with small number of samples
    - Works efficiently when there is a clear margin of separation between classes
    - Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient

    ## Disadvantages

    - Not suitable for large number of samples (training time is higher)
    - Not suitable for noisy data with overlapping classes

    ---
    """

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        n_iterations: int = 1000,
        lambda_: np.float64 = 0.01,
        x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None,
        debug: bool = True,
        copy_x: bool = True,
    ):
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

    def _y_hat(self, X: np.ndarray, W: np.ndarray, b: np.float64) -> np.ndarray:
        """
        ### Calculate the predicted value

        Parameters
        ----------

        `X` : np.ndarray
        - The input data of shape (n_samples, n_features)

        `W` : np.ndarray
        - The weights of shape (n_features,)

        `b` : np.float64
        - The intercept


        Returns
        -------

        `np.ndarray`
        - The predicted value of shape (n_samples, 1)

        ---
        """
        return np.dot(X, W) - b

    def _cost(
        self, X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64
    ) -> np.float64:
        """
        ### Calculate the cost of the model
        (Hinge loss)

        Parameters
        ----------

        `X` : np.ndarray
        - The input data of shape (n_samples, n_features)

        `y` : np.ndarray
        - The target data of shape (n_samples, 1)

        `W` : np.ndarray
        - The weights of shape (n_features,)

        `b` : np.float64
        - The intercept


        Returns
        -------

        `np.float64`
        - The cost

        ---
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
        ### Calculate the gradient of the model

        Parameters
        ----------

        `X` : np.ndarray
        - The input data of shape (n_samples, n_features)

        `y` : np.ndarray
        - The target data of shape (n_samples, 1)

        `W` : np.ndarray
        - The weights of shape (n_features,)

        `b` : np.float64
        - The intercept


        Returns
        -------

        `Union[np.ndarray, np.float64]`
        - The gradient of the model

        ---
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
        ### Get the X and y data

        Parameters
        ----------

        `X` : np.ndarray
        - The input data of shape (n_samples, n_features)

        `Y` : np.ndarray, optional
        - The target data of shape (n_samples, 1)


        Returns
        -------

        `Tuple[np.ndarray, np.ndarray]`
        - The X and Y data

        ---
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
        ### Fit the model to the data

        Parameters
        ----------

        `X` : np.ndarray
        - The input data of shape (n_samples, n_features)

        `y` : np.ndarray
        - The target data of shape (n_samples, 1) or (n_samples,)

        `W` : np.ndarray, optional
        - The weights of shape (n_features,), by default None
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

        assert X.shape[0] == Y.shape[0], "The number of samples must be equal"

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
        ### Predict the target data

        Parameters
        ----------

        `X` : np.ndarray
        - The input data of shape (n_samples, n_features)


        Returns
        -------

        `np.ndarray`
        - The predicted target data of shape (n_samples,)

        ---
        """
        X = self._validate_data(X)

        # Return 1 if the prediction is greater than or equal to 0, otherwise return -1
        return np.where(self._y_hat(X, self._weights, self._intercept) >= 0, 1, 0)

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
