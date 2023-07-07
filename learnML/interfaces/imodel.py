from abc import ABC, abstractmethod
import numpy as np
from typing import Union

from .ifeature_scaling import IFeatureScaling


class IModel(ABC):
    """Interface for model classes."""

    @abstractmethod
    def __init__(
        self,
        learning_rate: np.float64,
        num_iterations: int,
        X_scalar: IFeatureScaling,
        Y_scalar: IFeatureScaling,
        debug: bool,
        copy_X: bool,
    ) -> None:
        self._learning_rate = learning_rate
        self._num_iterations = num_iterations
        self._X_scalar = X_scalar
        self._Y_scalar = Y_scalar
        self._debug = debug
        self._copy_X = copy_X

        self._J_history: list = None
        self._p_history: list = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        w: Union[np.ndarray, np.float64],
        b: np.float64,
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The input array
        Y : np.ndarray
            The output array
        w : Union[np.ndarray, np.float64]
            The weight array
        b : np.float64
            The intercept

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output given the input.

        Parameters
        ----------
        X : np.ndarray
            The input array

        Returns
        -------
        np.ndarray
            The predicted output array
        """
        pass
