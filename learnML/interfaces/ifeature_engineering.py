from abc import ABC, abstractmethod
import numpy as np


class IFeatureEngineering(ABC):
    """Interface for feature engineering classes."""

    @abstractmethod
    def __init__(self, data: np.ndarray) -> None:
        """
        Parameters
        ----------
        data : np.ndarray
            The input array of shape (n_samples, n_features)
        """

        self._data = data

    @abstractmethod
    def fit(self, data: np.ndarray = None) -> None:
        """
        Fit the feature engineer to data

        Parameters
        ----------
        data : np.ndarray, optional
            The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        Transform data using feature engineer

        Parameters
        ----------
        data : np.ndarray, optional
            The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)

        Returns
        -------
        np.ndarray
            The transformed data of shape (n_samples, n_features)
        """
        pass

    @abstractmethod
    def fit_transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        Fit the feature engineer with data and transform with it

        Parameters
        ----------
        data : np.ndarray, optional
            The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)

        Returns
        -------
        np.ndarray
            The transformed data of shape (n_samples, n_features)
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Convert the data back to the original representation

        Parameters
        ----------
        data : np.ndarray
            The input array of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The transformed data of shape (n_samples, n_features)
        """
        pass
