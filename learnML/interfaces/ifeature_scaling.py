# from abc import ABC, abstractmethod
# import numpy as np
# from typing import Union


# class IFeatureScaling(ABC):
#     """Interface for feature scaling classes."""

#     @abstractmethod
#     def __init__(self, data: np.ndarray, index: Union[int, list, range]) -> None:
#         """
#         Parameters
#         ----------
#         data : np.ndarray
#             The data to be normalized
#         index : int, list, range, optional
#             The index of the columns to be normalized
#         """
#         self._data = data

#         if isinstance(index, int):
#             self._indexes = [index]
#         elif isinstance(index, list):
#             self._indexes = index
#         elif isinstance(index, range):
#             self._indexes = list(index)
#         else:
#             self._indexes = None

#     def fit_transform(self, data: np.ndarray = None) -> np.ndarray:
#         """
#         Normalize the data

#         Parameters
#         ----------
#         data : np.ndarray, optional
#             The data to be normalized, by default None

#         Returns
#         -------
#         np.ndarray
#             The normalized data
#         """
#         pass

#     def inverse_transform(self, data: np.ndarray = None) -> np.ndarray:
#         """
#         Inverse normalize the data

#         Parameters
#         ----------
#         data : np.ndarray, optional
#             The data to be inverse normalized, by default None

#         Returns
#         -------
#         np.ndarray
#             The inverse normalized data
#         """
#         pass
