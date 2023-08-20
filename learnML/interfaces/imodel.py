from abc import ABC, abstractmethod
import numpy as np


class IModel(ABC):
    """Interface for model classes."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The array like object containing the input
            data of shape (n_samples, n_features)
        Y : np.ndarray
            The array like object containing the output
            data of shape (n_samples, n_targets) or (n_samples,)

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
            The array like object containing the input
            data of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The array like object containing the output
            data of shape (n_samples, n_targets) or (n_samples,)
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate the score of the model.

        Parameters
        ----------
        X : np.ndarray
            The array like object containing the input
            data of shape (n_samples, n_features)
        Y : np.ndarray
            The array like object containing the output
            data of shape (n_samples, n_targets) or (n_samples,)

        Returns
        -------
        float
            The score of the model
        """
        pass
