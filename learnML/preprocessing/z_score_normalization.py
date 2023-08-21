from typing import Union
import numpy as np

from ..interfaces import IFeatureEngineering


class ZScoreNormalization(IFeatureEngineering):
    """
    # Z-Score Normalization

    This class provides Z-score normalization (also known as standardization) for input data. Z-score normalization transforms data so that it has a mean of 0 and a standard deviation of 1. This is achieved by subtracting the mean of each column and dividing by the standard deviation of each column.

    ---

    ## Mathematical Explanation

    Z-score normalization (standardization) is a statistical technique used to transform features in a dataset to have a mean of 0 and a standard deviation of 1. This transformation is applied column-wise to the features.

    For a given feature `X` with values `x_1, x_2, ..., x_n`, the Z-score normalization is calculated as follows:

    - Calculate the mean (`μ`) of the feature:

    ```
    μ = (x_1 + x_2 + ... + x_n) / n
    ```

    - Calculate the standard deviation (`σ`) of the feature:

    ```
    σ = sqrt(((x_1 - μ)^2 + (x_2 - μ)^2 + ... + (x_n - μ)^2) / n)
    ```

    - For each value `x_i` in the feature, compute the Z-score:

    ```
    z_i = (x_i - μ) / σ
    ```

    The Z-score normalization ensures that the transformed values have a mean of 0 and a standard deviation of 1, which is beneficial for many machine learning algorithms that assume standardized data. It also helps in comparing and visualizing features on a similar scale, avoiding potential issues caused by features with different scales.

    The `ZScoreNormalization` class provides an implementation of this mathematical process, allowing you to easily normalize your data using Z-score normalization.

    ---

    ## Usage

    To use the `ZScoreNormalization` class, follow the general steps below:

    1. Import the class from the `learnML.preprocessing` module
    2. Create an instance of the `ZScoreNormalization` class
    3. Call the `fit_transform` method to fit the data and transform it
    4. Call the `inverse_transform` method to inverse the transformation

    ```python
    from learnML.preprocessing import ZScoreNormalization

    # Create an instance of the ZScoreNormalization class
    z_score_normalization = ZScoreNormalization(data)

    # Fit and transform the data
    normalized_data = z_score_normalization.fit_transform(data)

    # Inverse the transformation
    inverse_data = z_score_normalization.inverse_transform(normalized_data)
    ```

    ---
    """

    def __init__(self, data: np.ndarray, index: Union[int, list, range] = None) -> None:
        """
        Parameters
        ----------

        `data` : np.ndarray
        - The data to be normalized

        `index` : int, list, range, optional
        - The index of the columns to be normalized
        - If `index` is an integer, then the column at that index will be normalized
        - If `index` is a list, then the columns at the indexes in the list will be normalized
        - If `index` is a range, then the columns at the indexes in the range will be normalized
        - If `index` is None, then all columns will be normalized

        ---
        """
        data = self.__get_numpy_array(data)
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
        ### Calculate the mean of each column

        Returns
        -------

        `np.ndarray`
        - The mean of each column

        ---
        """
        if self._mean is None:
            self._mean = np.mean(self._data, axis=0)
        return self._mean

    def _get_std(self) -> np.ndarray:
        """
        ### Calculate the standard deviation of each column

        Returns
        -------

        `np.ndarray`
        - The standard deviation of each column

        ---
        """
        if self._std is None:
            self._std = np.std(self._data, axis=0)
        return self._std

    def fit(self, data: np.ndarray = None) -> None:
        """
        ### Fit the data by calculating the mean and standard deviation of each column

        Parameters
        ----------

        `data` : np.ndarray, optional
        - The data to be normalized, by default None
        - If `data` is None, then the data passed to the constructor will be used
        - If `data` is not None, then the data passed to the constructor will be ignored

        ---
        """
        if data is None:
            data = self._data

        self._get_mean()
        self._get_std()

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        ### Transform the data by normalizing it

        Parameters
        ----------

        `data` : np.ndarray
        - The data to be normalized
        - The data must have the same number of columns as the data passed to the constructor

        Returns
        -------

        `np.ndarray`
        - The normalized data
        - The normalized data will have the same shape as the input data

        ---
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
        ### Fit and transform the data by normalizing it

        Parameters
        ----------

        `data` : np.ndarray, optional
        - The data to be normalized, by default None
        - If `data` is None, then the data passed to the constructor will be used
        - If `data` is not None, then the data passed to the constructor will be ignored

        Returns
        -------

        `np.ndarray`
        - The normalized data after fitting and transforming
        - The normalized data will have the same shape as the input data

        ---
        """
        if data is None:
            data = self._data
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        ### Inverse the data by denormalizing it

        Parameters
        ----------

        `data` : np.ndarray
        - The data to be inverse
        - The data must have the same number of columns as the data passed to the constructor

        Returns
        -------

        `np.ndarray`
        - The inverse data after denormalizing
        - The inverse data will have the same shape as the input data

        ---
        """
        if self._indexes is None:
            return np.multiply(data, self._get_std()) + self._get_mean()
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
