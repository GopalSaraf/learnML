from abc import ABC, abstractmethod
import numpy as np

from .imodel import IModel


class IRegression(IModel, ABC):
    """
    # IReression

    Interface for regression model classes.

    `IRegression` is an abstract interface that extends the `IModel` interface and specifies the methods and attributes expected from regression model classes. Regression models are used to predict continuous target variables based on input features.

    ---

    ## Usage

    ```python
    from learnML.interfaces import IRegression

    # Create a custom regression model class that implements the IRegression interface
    class CustomRegression(IRegression):
        def __init__(self, learning_rate, n_iterations):
            # Initialize necessary attributes or hyperparameters
            super().__init__(learning_rate, n_iterations)

        def fit(self, X, Y):
            # Implement the training logic for the regression model
            pass

        def predict(self, X):
            # Implement the prediction logic for the regression model
            pass

        def score(self, X, Y):
            # Implement the scoring logic for the regression model
            pass

    # Create an instance of the custom regression model
    model = CustomRegression(learning_rate=0.01, n_iterations=1000)

    # Load training data
    X_train = ...
    Y_train = ...

    # Fit the model to the training data
    model.fit(X_train, Y_train)

    # Load test data
    X_test = ...
    Y_test = ...

    # Make predictions using the trained model
    predictions = model.predict(X_test)

    # Evaluate the model's performance
    score = model.score(X_test, Y_test)

    # Get cost history and model parameters history
    cost_history = model.get_cost_history()
    params_history = model.get_parameter_history()

    # Get learned weights and intercept
    weights = model.get_weights()
    intercept = model.get_intercept()
    ```

    ---
    """

    def __init__(
        self,
        learning_rate: np.float64,
        n_iterations: int,
        debug: bool = True,
        copy_x: bool = True,
    ) -> None:
        """
        Parameters
        ----------

        `learning_rate` : np.float64
        - The learning rate

        `n_iterations` : int
        - The number of iterations

        `debug` : bool, optional
        - Whether to print debug messages, by default True

        `copy_x` : bool, optional
        - Whether to copy the input array, by default True

        ---
        """
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations

        self._debug = debug
        self._copy_x = copy_x

        self._weights: np.ndarray = None
        self._intercept: np.float64 = None

        self._cost_history: np.ndarray = None
        self._params_history: np.ndarray = None

        self._debug_freq = n_iterations // 10

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        return super().fit(X, Y)

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)

    @abstractmethod
    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return super().score(X, Y)

    def get_cost_history(self) -> np.ndarray:
        """
        ### Return the history of the cost function.

        Returns
        -------
        `np.ndarray`
        - The history of the cost function

        ---
        """
        return self._cost_history

    def get_parameter_history(self) -> np.ndarray:
        """
        ### Return the history of the parameters.

        Returns
        -------
        `np.ndarray`
        - The history of the parameters

        ---
        """
        return self._params_history

    def get_weights(self) -> np.ndarray:
        """
        ### Return the weights.

        Returns
        -------
        `np.ndarray`
        - The weights

        ---
        """
        return self._weights

    def get_intercept(self) -> np.float64:
        """
        ### Return the intercept.

        Returns
        -------
        `np.float64`
        - The intercept

        ---
        """
        return self._intercept

    def _debug_print(self, iteration: int, cost: np.float64) -> None:
        """
        ### Print the current iteration and cost.

        Parameters
        ----------
        `iteration` : int
        - The current iteration

        `cost` : np.float64
        - The current cost

        ---
        """
        n = len(str(self._n_iterations)) + 1
        print(f"Iteration: {iteration:{n}n} | Cost: {cost:0.6e}")
