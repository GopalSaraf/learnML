from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class IFeatureEngineering(ABC):
    """Interface for feature engineering classes."""

    @abstractmethod
    def __init__(self, data: np.ndarray, degree: Union[int, list]) -> None:
        """
        Parameters
        ----------
        data : np.ndarray
            The input array
        degree : Union[int, list], optional
            The degree of the polynomial, by default 2
            Either a single integer or a list of integers
            If a single integer is passed, then the polynomial features are generated from 2 to the given degree
            If a list of integers is passed, then the polynomial features are generated for the given degrees
        """

        self._data = data

        self._degrees = (
            degree if isinstance(degree, list) else list(range(2, degree + 1))
        )

    @abstractmethod
    def transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray, optional
            The input array, by default None (uses the input array passed in the constructor)

        Returns
        -------
        np.ndarray
            The polynomial features of the input array
        """
        pass
