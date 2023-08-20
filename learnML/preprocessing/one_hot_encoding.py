import numpy as np
from typing import Union

from ..interfaces import IFeatureEngineering


class OneHotEncoder(IFeatureEngineering):
    """One-hot encoding feature engineering class."""

    def __init__(
        self, data: np.ndarray, indexes: Union[int, list, range] = None
    ) -> None:
        """
        Parameters
        ----------
        data : np.ndarray
            The input array of shape (n_samples, n_features)
        indexes : int, list, range, optional
            The column indexes to one-hot encode
            If None, then all columns with dtype "O" are one-hot encoded, by default None
        """
        super().__init__(data)
        self._categories = None

        if isinstance(indexes, int):
            self._indexes = [indexes]
        elif isinstance(indexes, list):
            self._indexes = indexes
        elif isinstance(indexes, range):
            self._indexes = list(indexes)
        else:
            self._indexes = self._identify_categorical_columns()

    def _identify_categorical_columns(self):
        """
        Automatically identify categorical columns based on their data types.

        Returns
        -------
        list
            The column indexes of the categorical columns.
        """
        categorical_indexes = []
        for idx in range(self._data.shape[1]):
            if np.issubdtype(self._data[:, idx].dtype, np.str_):
                categorical_indexes.append(idx)
        return categorical_indexes

    def fit(self, data: np.ndarray = None) -> None:
        """
        Fit the feature engineer to data

        Parameters
        ----------
        data : np.ndarray, optional
            The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)
        """

        if data is None:
            data = self._data

        self._categories = [np.unique(data[:, i]) for i in self._indexes]

    def _get_encoding_string(self, num: int, num_bits: int) -> str:
        """
        Get the binary encoding string for a number

        Parameters
        ----------
        num : int
            The number to encode
        num_bits : int
            The number of bits to use for the encoding

        Returns
        -------
        str
            The binary encoding string for the number
        """
        encoding_string = ""
        for i in range(num_bits):
            if i == num:
                encoding_string += "1"
            else:
                encoding_string += "0"
        return encoding_string

    def _get_encodings(self):
        """
        Get the encodings for the categorical columns

        Returns
        -------
        list
            The encodings for the categorical columns
        """
        # Should return list of dicts with keys as categories and values as encodings

        encodings = []
        for i in range(len(self._categories)):
            num_bits = len(self._categories[i])
            encodings.append(
                {
                    self._categories[i][j]: self._get_encoding_string(j, num_bits)
                    for j in range(len(self._categories[i]))
                }
            )
        return encodings

    def transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        Transform the data

        Parameters
        ----------
        data : np.ndarray, optional
            The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)

        Returns
        -------
        np.ndarray
            The transformed array of shape (n_samples, n_features)
        """

        if data is None:
            data = self._data

        m, n = data.shape
        extra_columns = 0

        for i in range(len(self._categories)):
            extra_columns += len(self._categories[i]) - 1

        transformed_data = np.zeros((m, n + extra_columns), dtype=data.dtype)
        encodings = self._get_encodings()

        col_shift = 0
        cat_ind = 0

        for i in range(n):
            if i in self._indexes:
                for j in range(m):
                    for k in range(len(self._categories[cat_ind])):
                        transformed_data[j][i + col_shift] = np.float64(
                            encodings[cat_ind][data[j][i]][k]
                        )
                        col_shift += 1
                    col_shift -= len(self._categories[cat_ind])
                col_shift += len(self._categories[cat_ind]) - 1
                cat_ind += 1
            else:
                transformed_data[:, i + col_shift] = data[:, i]

        try:
            transformed_data = transformed_data.astype(np.float64)
        except ValueError:
            pass

        return transformed_data

    def inverse_transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        Inverse transform the data

        Parameters
        ----------
        data : np.ndarray, optional
            The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)

        Returns
        -------
        np.ndarray
            The inverse transformed array of shape (n_samples, n_features)
        """

        if data is None:
            data = self._data

        m, n = data.shape
        extra_columns = 0

        for i in range(len(self._categories)):
            extra_columns += len(self._categories[i]) - 1

        inverse_transformed_data = np.zeros((m, n - extra_columns), dtype=data.dtype)
        encodings = self._get_encodings()

        col_shift = 0
        cat_ind = 0

        for i in range(n):
            if i in self._indexes:
                for j in range(m):
                    encoding = ""
                    for k in range(len(self._categories[cat_ind])):
                        encoding += str(data[j][i + col_shift])
                        col_shift += 1
                    col_shift -= len(self._categories[cat_ind])
                    inverse_transformed_data[j][i] = np.float64(
                        list(encodings[cat_ind].keys())[
                            list(encodings[cat_ind].values()).index(encoding)
                        ]
                    )
                col_shift += len(self._categories[cat_ind]) - 1
                cat_ind += 1
            else:
                inverse_transformed_data[:, i] = data[:, i + col_shift]

        try:
            inverse_transformed_data = inverse_transformed_data.astype(np.float64)
        except ValueError:
            pass

        return inverse_transformed_data

    def fit_transform(self, data: np.ndarray = None) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def get_categories(self):
        return self._categories

    def get_encodings(self):
        encoding_maps = {}
        encodings = self._get_encodings()

        for i in range(len(self._categories)):
            for j in range(len(self._categories[i])):
                encoding_maps[self._categories[i][j]] = np.array(
                    list(list(encodings[i].values())[j]), dtype=np.float64
                )

        return encoding_maps
