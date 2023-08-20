import numpy as np
from typing import Tuple, Union

from ..interfaces import IFeatureEngineering
from .linear_regression import LinearRegression
from ..preprocessing import PolynomialFeatures


class PolynomialRegression(LinearRegression):
    """Polynomial Linear Regression Model"""

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        n_iterations: int = 1000,
        degree: Union[int, list] = 2,
        lambda_: np.float64 = 0,
        x_scalar: Union[IFeatureEngineering, list] = None,
        y_scalar: Union[IFeatureEngineering, list] = None,
        debug: bool = True,
        copy_x: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        learning_rate : np.float64, optional
            The learning rate, by default 0.001
        n_iterations : int, optional
            The number of iterations, by default 1000
        degree : Union[int, list], optional
            The degree of the polynomial, by default 2
        lambda_ : np.float64, optional
            The regularization parameter, by default 0
        x_scalar : {IFeatureEngineering, list(IFeatureEngineering)}, optional
            The feature engineering for the input data, by default None
        y_scalar : {IFeatureEngineering, list(IFeatureEngineering)}, optional
            The feature engineering for the output data, by default None
        debug : bool, optional
            Whether to print debug messages, by default True
        copy_x : bool, optional
            Whether to copy the input array, by default True

        Degree
        ------
        It can be a single integer, a list of integers or a dictionary of integers and lists of integers

            If it is a single integer, then the polynomial features of all the features of the input array will be generated with the given degree from 1 to the given degree

            If it is a list of integers, then the polynomial features of the features of the input array will be generated with the given degrees in the list

            If it is a dictionary of integers and lists of integers, then for each key-value pair in the dictionary, the polynomial features of the features at the key index of the input array will be generated with the given degrees in the list

            Examples:
                degree = 2
                    All the features of the input array will be used to generate the polynomial features with degree 0, 1 and 2

                degree = [2, 3]
                    All the features of the input array will be used to generate the polynomial features with degree 2 and 3

                degree = {0: [2, 3], 1: 2}
                    The first feature of the input array will be used to generate the polynomial features with degree 2 and 3
                    The second feature of the input array will be used to generate the polynomial features with degree 0, 1 and 2
        """
        super().__init__(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            lambda_=lambda_,
            x_scalar=x_scalar,
            y_scalar=y_scalar,
            debug=debug,
            copy_x=copy_x,
        )

        self._degree = degree
        self._polynomial_features: PolynomialFeatures = None

    def _get_polynomial(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray
            The input array of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The polynomial of the given degree of shape (n_samples, n_features * degree)
        """
        if self._polynomial_features is None:
            self._polynomial_features = PolynomialFeatures(
                data=data, degree=self._degree
            )

        return self._polynomial_features.transform(data)

    def _validate_data(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the input and output arrays.

        Parameters
        ----------
        X : np.ndarray
            The input array of shape (n_samples, n_features)
        Y : np.ndarray, optional
            The output array of shape (n_samples,) or (n_samples, 1)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The input and output arrays
        """
        if Y is not None:
            assert (
                X.shape[0] == Y.shape[0]
            ), "X and Y must have the same number of samples"

        if self._copy_x:
            X = np.copy(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for scalar in self._x_scalar:
            X = scalar.transform(X)

        if Y is not None:
            if Y.ndim == 2:
                Y = Y.reshape(-1)

            for scalar in self._y_scalar:
                Y = scalar.transform(Y)

        X = self._get_polynomial(data=X)

        if Y is None:
            return X
        return X, Y
