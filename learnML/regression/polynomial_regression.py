import copy
import numpy as np
from typing import Tuple, Union, Dict, List

from ..interfaces import IFeatureEngineering
from .linear_regression import LinearRegression
from ..preprocessing import PolynomialFeatures


class PolynomialRegression(LinearRegression):
    """
    # Polynomial Linear Regression Model

    Polynomial Regression is an extension of Linear Regression that allows for the modeling of nonlinear relationships between the input features and the target variable. It achieves this by introducing polynomial features, which are derived from raising the original input features to various powers. This approach can capture more complex patterns in the data and provide a higher degree of flexibility in modeling.

    ---

    ## Mathematical Approach

    Polynomial Regression aims to approximate the relationship between the input feature `x` and the target variable `y` using a polynomial equation of the form:

    ```
    y = b0 + b1*x + b2*x^2 + ... + bn*x^n
    ```

    Where:

    - `y` is the predicted output (target variable).
    - `x` is the input feature.
    - `b0, b1, ..., bn` are the coefficients of the polynomial terms.
    - `n` is the degree of the polynomial.

    The degree `n` determines the complexity of the polynomial curve. By increasing the degree, the model can fit the training data more closely, but it might also lead to overfitting.

    Polynomial Regression is implemented using a linear regression model by treating the polynomial terms as separate input features. The model learns the optimal coefficients `b0, b1, ..., bn` that minimize the difference between predicted values and actual target values.

    ---

    ## Usage

    To utilize the Polynomial Regression model, follow these steps:

    1. Import the `PolynomialRegression` class from the appropriate module.
    2. Create an instance of the `PolynomialRegression` class, specifying hyperparameters such as learning rate, degree, etc.
    3. Fit the model to your training data using the `fit` method.
    4. Make predictions on new data using the `predict` method.
    5. Evaluate the model's performance using the `score` method.

    ```python
    from learnML.regression import PolynomialRegression

    # Create an instance of PolynomialRegression
    model = PolynomialRegression(learning_rate=0.01, degree=2, n_iterations=1000)

    # Fit the model to training data
    model.fit(X_train, Y_train)

    # Make predictions on new data
    predictions = model.predict(X_test)

    # Calculate the model's score
    model_score = model.score(X_test, Y_test)
    ```

    ---
    """

    def __init__(
        self,
        learning_rate: np.float64 = 0.001,
        n_iterations: int = 1000,
        degree: Union[int, List[int], Dict[int, Union[int, List[int]]]] = 2,
        lambda_: np.float64 = 0,
        x_scalar: Union[IFeatureEngineering, list] = None,
        y_scalar: Union[IFeatureEngineering, list] = None,
        debug: bool = True,
        copy_x: bool = True,
    ) -> None:
        """
        Parameters
        ----------

        `learning_rate` : np.float64, optional
        - The learning rate, by default 0.001
        - The learning rate determines how much the weights are updated at each iteration
        - A low learning rate will take longer to converge, but a high learning rate may overshoot the optimal solution

        `n_iterations` : int, optional
        - The number of iterations, by default 1000
        - The number of iterations determines how many times the weights are updated
        - A higher number of iterations will take longer to converge, but a lower number of iterations may not be enough to converge

        `degree` : Union[int, List[int], Dict[int, Union[int, List[int]]]], optional
        - The degree of the polynomial, by default 2
        - It can be a single integer, a list of integers or a dictionary of integers and lists of integers
        - If it is a single integer, then the polynomial features of all the features of the input array will be generated with the given degree from 1 to the given degree
        - If it is a list of integers, then the polynomial features of all the features of the input array will be generated with the given degrees
        - If it is a dictionary of integers and lists of integers, then for each key-value pair in the dictionary, the polynomial features of the features at the key index of the input array will be generated with the given degrees in the list

        `lambda_` : np.float64, optional
        - The regularization parameter, by default 0
        - The regularization parameter helps prevent overfitting by penalizing large weights
        - A higher regularization parameter will penalize large weights more, but a lower regularization parameter may not be enough to prevent overfitting

        `x_scalar` : Union[IFeatureEngineering, List[IFeatureEngineering]], optional
        - The feature engineering for the input data, by default None
        - If a list is provided, the feature engineering will be applied in the order provided
        - If a single feature engineering is provided, it will be applied to all input data

        `y_scalar` : Union[IFeatureEngineering, List[IFeatureEngineering]], optional
        - The feature engineering for the output data, by default None
        - If a list is provided, the feature engineering will be applied in the order provided
        - If a single feature engineering is provided, it will be applied to all output data

        `debug` : bool, optional
        - Whether to print debug messages, by default True
        - Debug messages include the cost at each iteration

        `copy_x` : bool, optional
        - Whether to copy the input array, by default True
        - If False, the input array will be overwritten


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
        ### Return the polynomial of the given degree

        Parameters
        ----------

        `data` : np.ndarray
        - The input array of shape (n_samples, n_features)


        Returns
        -------

        `np.ndarray`
        - The polynomial of the given degree of shape (n_samples, n_features * degree)

        ---
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
        ### Return the input and output arrays.

        Parameters
        ----------

        `X` : np.ndarray
        - The input array of shape (n_samples, n_features)

        `Y` : np.ndarray, optional
        - The output array of shape (n_samples,) or (n_samples, 1)


        Returns
        -------

        `Tuple[np.ndarray, np.ndarray]`
        - The input and output arrays

        ---
        """
        if self._copy_x:
            X = copy.deepcopy(X)

        X = self.__get_numpy_array(X)
        Y = self.__get_numpy_array(Y) if Y is not None else None

        if Y is not None:
            assert (
                X.shape[0] == Y.shape[0]
            ), "X and Y must have the same number of samples"

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
