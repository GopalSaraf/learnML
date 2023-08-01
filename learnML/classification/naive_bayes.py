import numpy as np
from typing import Tuple, Union

from ..interfaces import IModel


class NaiveBayesClassifier(IModel):
    """Naive Bayes Classifier Model"""

    def __init__(self, debug: bool = True) -> None:
        """
        Parameters
        ----------
        debug : bool, optional
            Whether to print debug messages, by default True
        """
        self._debug = debug

        self._classes: np.ndarray = None
        self._class_counts: np.ndarray = None
        self._class_probs: np.ndarray = None
        self._feature_counts: np.ndarray = None
        self._feature_probs: np.ndarray = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)
        Y : np.ndarray
            The output data of shape (n_samples,)
        """
        self._classes, self._class_counts = np.unique(Y, return_counts=True)
        self._class_probs = self._class_counts / Y.shape[0]

        self._feature_counts = np.zeros((self._classes.shape[0], X.shape[1]))
        for i, c in enumerate(self._classes):
            self._feature_counts[i] = X[Y == c].sum(axis=0)

        self._feature_probs = self._feature_counts / self._class_counts.reshape(-1, 1)

        if self._debug:
            print("Classes:", self._classes)
            print("Class Counts:", self._class_counts)
            print("Class Probabilities:", self._class_probs)
            print("Feature Counts:\n", self._feature_counts)
            print("Feature Probabilities:\n", self._feature_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output given the input.

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The predicted output of shape (n_samples,)
        """
        return np.apply_along_axis(self._predict, 1, X)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output given the input.

        Parameters
        ----------
        x : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The predicted output
        """
        probs = self._class_probs * np.prod(
            self._feature_probs**x * (1 - self._feature_probs) ** (1 - x), axis=1
        )
        return self._classes[np.argmax(probs)]
