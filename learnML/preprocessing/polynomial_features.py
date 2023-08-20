import numpy as np
from typing import Union, Dict, List


class PolynomialFeatures:
    """Polynomial Feature Generator Class"""

    def __init__(
        self,
        data: np.ndarray,
        degree: Union[int, List[int], Dict[int, Union[int, List[int]]]] = 2,
    ) -> None:
        """
        Parameters
        ----------
        data : np.ndarray
            The input array of shape (n_samples, n_features)
        degree : Union[int, List[int], Dict[int, Union[int, List[int]]]], optional
            The degree of the polynomial, by default 2

            It can be a single integer, a list of integers or a dictionary of integers and lists of integers

            If it is a single integer, then the polynomial features of all the features of the input array will be generated with the given degree from 1 to the given degree

            If it is a list of integers, then the polynomial features of the features of the input array will be generated with the given degrees in the list

            If it is a dictionary of integers and lists of integers, then for each key-value pair in the dictionary, the polynomial features of the features at the key index of the input array will be generated with the given degrees in the list

            Examples:
                degree = 2
                    All the features of the input array will be used to generate the polynomial features with degree 1 and 2

                degree = [2, 3]
                    All the features of the input array will be used to generate the polynomial features with degree 2 and 3

                degree = {0: [2, 3], 1: 2}
                    The first feature of the input array will be used to generate the polynomial features with degree 2 and 3
                    The second feature of the input array will be used to generate the polynomial features with degree 1 and 2
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
        data : np.float64
            The input value
        feature_idx : int
            The index of the feature
        degree : int
            The degree of the polynomial

        Returns
        -------
        np.ndarray
            The polynomial of the given degree for the specified feature
        """

        return data[:, feature_idx] ** degree

    def transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray, optional
            The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)

        Returns
        -------
        np.ndarray
            The polynomial features of the input array of shape (n_samples, n_features)
        """

        if data is None:
            data = self._data

        polynomial_features = [
            self._get_degree(data, i, degree)
            for i, degrees in self._degrees.items()
            for degree in degrees
        ]

        return np.column_stack([data, *polynomial_features])
