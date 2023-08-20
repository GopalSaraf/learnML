from abc import ABC, abstractmethod
import numpy as np

from .imodel import IModel


class IRegression(IModel, ABC):
    """Interface for regression model classes."""

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
        learning_rate : np.float64
            The learning rate
        n_iterations : int
            The number of iterations
        debug : bool, optional
            Whether to print debug messages, by default True
        copy_x : bool, optional
            Whether to copy the input array, by default True
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
        Return the history of the cost function.

        Returns
        -------
        np.ndarray
            The history of the cost function
        """
        return self._cost_history

    def get_parameter_history(self) -> np.ndarray:
        """
        Return the history of the parameters.

        Returns
        -------
        np.ndarray
            The history of the parameters
        """
        return self._params_history

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

    def _debug_print(self, iteration: int, cost: np.float64) -> None:
        """
        Print the current iteration and cost.

        Parameters
        ----------
        iteration : int
            The current iteration
        cost : np.float64
            The current cost
        """
        n = len(str(self._n_iterations)) + 1
        print(f"Iteration: {iteration:{n}n} | Cost: {cost:0.6e}")
