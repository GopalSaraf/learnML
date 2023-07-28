import numpy as np
from typing import Union

from ..interfaces import IFeatureEngineering


class PolynomialFeatures(IFeatureEngineering):
    """Class for generating polynomial features."""

    def __init__(self, data: np.ndarray, degree: Union[int, list] = 2) -> None:
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

        Examples
        --------
        >>> import numpy as np
        >>> from Preprocessing.feature_engg import FeatureEngineering
        >>> X = np.arange(0, 20, 1)
        >>> feature_engineering = FeatureEngineering(X, degree=5)
        >>> X_poly = feature_engineering.transform()
        >>> X_poly.shape
        (20, 5)
        """

        super().__init__(data, degree)

    def _get_degree(self, degree: int) -> np.ndarray:
        """
        Parameters
        ----------
        degree : int
            The degree of the polynomial

        Returns
        -------
        np.ndarray
            The polynomial of the given degree
        """

        return self._data**degree

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

        if data is None:
            data = self._data

        polynomial_features = np.hstack(
            [self._get_degree(degree) for degree in self._degrees]
        )

        return np.hstack([data, polynomial_features])
