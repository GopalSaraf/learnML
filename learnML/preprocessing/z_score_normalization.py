from typing import Union
import numpy as np

from ..interfaces import IFeatureEngineering


class ZScoreNormalization(IFeatureEngineering):
    """Z-Score Normalization (Standardization)"""

    def __init__(self, data: np.ndarray, index: Union[int, list, range] = None) -> None:
        """
        Parameters
        ----------
        data : np.ndarray
            The data to be normalized
        index : int, list, range, optional
            The index of the columns to be normalized
        """
        super().__init__(data)

        if isinstance(index, int):
            self._indexes = [index]
        elif isinstance(index, list):
            self._indexes = index
        elif isinstance(index, range):
            self._indexes = list(index)
        else:
            self._indexes = None

        self._mean = None
        self._std = None

    def _get_mean(self) -> np.ndarray:
        """
        Calculate the mean of each column

        Returns
        -------
        np.ndarray
            The mean of each column
        """
        if self._mean is None:
            self._mean = np.mean(self._data, axis=0)
        return self._mean

    def _get_std(self) -> np.ndarray:
        """
        Calculate the standard deviation of each column

        Returns
        -------
        np.ndarray
            The standard deviation of each column
        """
        if self._std is None:
            self._std = np.std(self._data, axis=0)
        return self._std

    def fit(self, data: np.ndarray = None) -> None:
        """
        Fit the data

        Parameters
        ----------
        data : np.ndarray, optional
            The data to be normalized, by default None
        """
        if data is None:
            data = self._data

        self._get_mean()
        self._get_std()

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data

        Parameters
        ----------
        data : np.ndarray
            The data to be normalized

        Returns
        -------
        np.ndarray
            The normalized data
        """
        if self._indexes is None:
            return (data - self._get_mean()) / self._get_std()
        else:
            normalized_data = np.zeros(data.shape)
            normalized_data[:, self._indexes] = (
                data[:, self._indexes] - self._get_mean()[self._indexes]
            ) / self._get_std()[self._indexes]
            normalized_data[
                :, [i for i in range(data.shape[1]) if i not in self._indexes]
            ] = data[:, [i for i in range(data.shape[1]) if i not in self._indexes]]
            return normalized_data

    def fit_transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        Fit and transform the data

        Parameters
        ----------
        data : np.ndarray, optional
            The data to be normalized, by default None

        Returns
        -------
        np.ndarray
            The normalized data
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse the data

        Parameters
        ----------
        data : np.ndarray
            The data to be inverse

        Returns
        -------
        np.ndarray
            The inverse data
        """
        if self._indexes is None:
            return data * self._get_std() + self._get_mean()
        else:
            inverse_data = np.zeros(data.shape)
            inverse_data[:, self._indexes] = (
                data[:, self._indexes] * self._get_std()[self._indexes]
                + self._get_mean()[self._indexes]
            )
            inverse_data[
                :, [i for i in range(data.shape[1]) if i not in self._indexes]
            ] = data[:, [i for i in range(data.shape[1]) if i not in self._indexes]]
            return inverse_data
