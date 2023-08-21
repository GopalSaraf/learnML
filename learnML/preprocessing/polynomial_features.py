import numpy as np
from typing import Union, Dict, List


class PolynomialFeatures:
    """
    # Polynomial Feature Generator Class

    The `PolynomialFeatures` class is designed to generate polynomial features from input data. Polynomial features are derived from the original features by raising them to various degrees. This process can help capture non-linear relationships between features and improve the performance of machine learning algorithms.

    ---

    ## Mathematical Explanation

    Polynomial features are derived from the original features by raising them to various degrees. For example, if we have a feature `x` with values `x_1, x_2, ..., x_n`, then the polynomial features of `x` with degree 2 are `x_1^2, x_2^2, ..., x_n^2`. The polynomial features of `x` with degree 3 are `x_1^3, x_2^3, ..., x_n^3`, and so on.

    The `PolynomialFeatures` class provides an implementation of this mathematical process, allowing you to easily generate polynomial features from your data.

    ---

    ## Usage

    To use the `PolynomialFeatures` class, follow the general steps below:

    1. Import the class from the `learnML.preprocessing` module
    2. Create an instance of the `PolynomialFeatures` class
    3. Call the `transform` method to transform the data

    ```python
    from learnML.preprocessing import PolynomialFeatures

    # Create an instance of the PolynomialFeatures class
    polynomial_features = PolynomialFeatures(data, degree)

    # Transform the data
    polynomial_features.transform()
    ```

    ---
    """

    def __init__(
        self,
        data: np.ndarray,
        degree: Union[int, List[int], Dict[int, Union[int, List[int]]]] = 2,
    ) -> None:
        """
        Parameters
        ----------

        `data` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `degree` : Union[int, List[int], Dict[int, Union[int, List[int]]]], optional
        - The degree of the polynomial, by default 2
        - It can be a single integer, a list of integers or a dictionary of integers and lists of integers
        - If it is a single integer, then the polynomial features of all the features of the input array will be generated with the given degree from 1 to the given degree
        - If it is a list of integers, then the polynomial features of all the features of the input array will be generated with the given degrees
        - If it is a dictionary of integers and lists of integers, then for each key-value pair in the dictionary, the polynomial features of the features at the key index of the input array will be generated with the given degrees in the list

        Degree
        ------

        Examples:

        ```python
        degree = 2
        # Generate polynomial features of degree 1 and 2 for all the features

        degree = [2, 3, 6]
        # Generate polynomial features of degree 2, 3 and 6 for all the features

        degree = {0: [2, 3, 6], 1: 2}
        # Generate polynomial features of degree 2, 3 and 6 for the first feature
        # Generate polynomial features of degree 1 and 2 for the second feature
        ```

        ---
        """
        data = data if data.ndim == 2 else data.reshape(-1, 1)
        self._data = data

        if isinstance(degree, int):
            self._degrees = {
                i: list(range(2, degree + 1)) for i in range(data.shape[1])
            }
        elif isinstance(degree, list):
            self._degrees = {i: degree for i in range(data.shape[1])}
        elif isinstance(degree, dict):
            degree_dict = {}
            for i, degrees in degree.items():
                if isinstance(degrees, int):
                    degree_dict[i] = list(range(2, degrees + 1))
                elif isinstance(degrees, list):
                    degree_dict[i] = degrees
                else:
                    raise TypeError(f"Unsupported type for degrees: {type(degrees)}")
            self._degrees = degree_dict

    def _get_degree(
        self, data: np.float64, feature_idx: int, degree: int
    ) -> np.ndarray:
        """
        Parameters
        ----------

        `data` : np.float64
        - The input value

        `feature_idx` : int
        - The index of the feature

        `degree` : int
        - The degree of the polynomial


        Returns
        -------

        `np.ndarray`
        - The polynomial of the given degree for the specified feature

        ---
        """

        return data[:, feature_idx] ** degree

    def transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        ### Transform the input array into polynomial features

        Parameters
        ----------

        `data` : np.ndarray, optional
        - The input array of shape (n_samples, n_features), by default None
        - If None, then the input array passed to the constructor will be used
        - If not None, then the input array passed to the constructor will be ignored

        Returns
        -------

        `np.ndarray`
        - The polynomial features of the input array of shape (n_samples, n_features)

        ---
        """

        if data is None:
            data = self._data

        polynomial_features = [
            self._get_degree(data, i, degree)
            for i, degrees in self._degrees.items()
            for degree in degrees
        ]

        return np.column_stack([data, *polynomial_features])
