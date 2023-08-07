# import numpy as np
# from typing import Union

# from ..interfaces import IFeatureEngineering


# class PolynomialFeatures(IFeatureEngineering):
#     """Polynomial Feature Generator Class"""

#     def __init__(self, data: np.ndarray, degree: Union[int, list] = 2) -> None:
#         """
#         Parameters
#         ----------
#         data : np.ndarray
#             The input array of shape (n_samples, n_features)
#         degree : Union[int, list], optional
#             The degree of the polynomial, by default 2
#             Either a single integer or a list of integers
#             If a single integer is passed, then the polynomial features are generated from 2 to the given degree
#             If a list of integers is passed, then the polynomial features are generated for the given degrees
#         """

#         super().__init__(data)

#         self._degrees = (
#             degree if isinstance(degree, list) else list(range(2, degree + 1))
#         )

#     def _get_degree(self, degree: int) -> np.ndarray:
#         """
#         Parameters
#         ----------
#         degree : int
#             The degree of the polynomial

#         Returns
#         -------
#         np.ndarray
#             The polynomial of the given degree
#         """

#         return self._data**degree

#     def transform(self, data: np.ndarray = None) -> np.ndarray:
#         """
#         Parameters
#         ----------
#         data : np.ndarray, optional
#             The input array of shape (n_samples, n_features),
#             by default None (uses the input array passed in the constructor)

#         Returns
#         -------
#         np.ndarray
#             The polynomial features of the input array of shape (n_samples, n_features)
#         """

#         if data is None:
#             data = self._data

#         polynomial_features = np.hstack(
#             [self._get_degree(degree) for degree in self._degrees]
#         )

#         return np.hstack([data, polynomial_features])
