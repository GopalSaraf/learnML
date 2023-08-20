# Class [`PolynomialRegression`](/learnML/regression/polynomial_regression.py#L9)

## Introduction

Polynomial Regression is an extension of Linear Regression that allows for the modeling of nonlinear relationships between the input features and the target variable. It achieves this by introducing polynomial features, which are derived from raising the original input features to various powers. This approach can capture more complex patterns in the data and provide a higher degree of flexibility in modeling.

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

## Implementation

### Constructor

#### [`__init__(learning_rate: np.float64 = 0.001, n_iterations: int = 1000, degree: Union[int, list] = 2, lambda_: np.float64 = 0, x_scalar: Union[IFeatureEngineering, list] = None, y_scalar: Union[IFeatureEngineering, list] = None, debug: bool = True, copy_x: bool = True) -> None`](/learnML/regression/polynomial_regression.py#L12)

Initialize the Polynomial Regression model.

- `learning_rate` (np.float64, optional): The learning rate controls the step size during optimization.
- `n_iterations` (int, optional): The number of iterations for optimization.
- `degree` (Union[int, list], optional): The degree of the polynomial.
- `lambda_` (np.float64, optional): The regularization parameter.
- `x_scalar` (Union[IFeatureEngineering, list], optional): Feature engineering for input data.
- `y_scalar` (Union[IFeatureEngineering, list], optional): Feature engineering for output data.
- `debug` (bool, optional): Whether to print debug messages during training.
- `copy_x` (bool, optional): Whether to make a copy of the input array to avoid modifying the original data.

### Methods

#### [`_get_polynomial(self, data: np.ndarray) -> np.ndarray`](/learnML/regression/polynomial_regression.py#L77)

Get the polynomial features of the given degree.

- `data` (np.ndarray): The input array of shape `(n_samples, n_features)`.

The `_get_polynomial` method returns the polynomial features of the input data based on the specified degree.

#### [`_validate_data(self, X: np.ndarray, Y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]`](/learnML/regression/polynomial_regression.py#L96)

Validate and preprocess input and output data.

- `X` (np.ndarray): The input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray or None, optional): The output array of shape `(n_samples,)` or `(n_samples, 1)`.

The `_validate_data` method preprocesses input and output data, including applying feature engineering and generating polynomial features.

## Notes

- The `PolynomialRegression` class extends the `LinearRegression` class to perform Polynomial Regression with the flexibility of polynomial features.
- Polynomial Regression introduces polynomial terms to capture nonlinear relationships in data.
- The degree of the polynomial controls the complexity of the model, and careful selection is essential to avoid overfitting.
- Feature engineering can be applied using the `x_scalar` and `y_scalar` parameters.
- Private methods `_get_polynomial` and `_validate_data` are used for different aspects of the Polynomial Regression algorithm.
