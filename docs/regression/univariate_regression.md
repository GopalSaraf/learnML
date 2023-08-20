# Class [`UnivariateLinearRegression`](/learnML/regression/univariate_regression.py#L7)

## Introduction

Linear regression is a fundamental supervised machine learning algorithm that is used for predictive modeling when the relationship between a dependent variable and a single independent variable can be approximated as a linear equation. It is a versatile and commonly used technique in various fields for tasks such as predicting housing prices, sales projections, and more.

Linear regression aims to establish a linear relationship between the input variable (often referred to as the "feature" or "independent variable") and the output variable (referred to as the "target" or "dependent variable"). This relationship is represented by a linear equation of the form:

```
y = mx + b
```

Where:

- `y` is the predicted output (target variable).
- `x` is the input feature.
- `m` is the slope of the line (weight).
- `b` is the y-intercept.

The linear regression model learns the optimal values of `m` and `b` from the training data, aiming to minimize the difference between the predicted values and the actual target values.

## Usage

```python
from learnML.regression import UnivariateLinearRegression

# Create an instance of UnivariateLinearRegression
model = UnivariateLinearRegression(learning_rate=0.01, n_iterations=1000)

# Fit the model to data
model.fit(X_train, Y_train)

# Make predictions using the model
predictions = model.predict(X_test)

# Calculate the model's score
model_score = model.score(X_test, Y_test)
```

## Implementation

### Constructor

#### [`__init__(learning_rate: np.float64 = 0.001, n_iterations: int = 1000, x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, y_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, debug: bool = True, copy_x: bool = True) -> None`](/learnML/regression/univariate_regression.py#L24)

Initialize the Univariate Linear Regression model.

- `learning_rate` (np.float64, optional): The learning rate controls the step size of the gradient descent optimization process.
- `n_iterations` (int, optional): The number of iterations for the gradient descent optimization.
- `x_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for the input data to preprocess or scale the features.
- `y_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for the output data to preprocess or scale the target.
- `debug` (bool, optional): Whether to print debug information during training.
- `copy_x` (bool, optional): Whether to create a copy of the input data to avoid modifying the original data.

### Methods

#### [`fit(X: np.ndarray, Y: np.ndarray, w: np.float64 = 0.0, b: np.float64 = 0.0) -> None`](/learnML/regression/univariate_regression.py#L206)

Train the Univariate Linear Regression model.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray): The output array of shape `(n_samples,)`.
- `w` (np.float64, optional): The initial weight.
- `b` (np.float64, optional): The initial intercept.

#### [`predict(X: Union[np.ndarray, np.float64]) -> Union[np.ndarray, np.float64]`](/learnML/regression/univariate_regression.py#L269)

Predict the output using the trained model.

- `X` (Union[np.ndarray, np.float64]): The input value or array of shape `(n_samples,)`.

#### [`score(X: np.ndarray, Y: np.ndarray, w: np.float64 = None, b: np.float64 = None) -> np.float64`](/learnML/regression/univariate_regression.py#L310)

Calculate the cost function given input and output data.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray): The output array of shape `(n_samples,)`.
- `w` (np.float64, optional): The weight.
- `b` (np.float64, optional): The intercept.

### Private Methods

#### [`_y_hat(x: np.float64, w: np.float64, b: np.float64) -> np.float64`](/learnML/regression/univariate_regression.py#L68)

Calculate the predicted value given `x`, weight `w`, and intercept `b`.

- `x` (np.float64): The input value.
- `w` (np.float64): The weight.
- `b` (np.float64): The intercept.

#### [`_cost(X: np.ndarray, Y: np.ndarray, w: np.float64, b: np.float64) -> np.float64`](/learnML/regression/univariate_regression.py#L88)

Calculate the cost function (Mean Squared Error) given input `X`, target `Y`, weight `w`, and intercept `b`.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray): The target array of shape `(n_samples,)`.
- `w` (np.float64): The weight.
- `b` (np.float64): The intercept.

#### [`_gradient(X: np.ndarray, Y: np.ndarray) -> Tuple[np.float64, np.float64]`](/learnML/regression/univariate_regression.py#L123)

Calculate the gradient of the cost function with respect to weight and intercept.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray): The target array of shape `(n_samples,)`.

#### [`_validate_data(X: np.ndarray, Y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]`](/learnML/regression/univariate_regression.py#L156)

Validate and preprocess input and output data.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray or None, optional): The target array of shape `(n_samples,)`, default is `None`.

## Notes

- The `UnivariateLinearRegression` class performs linear regression on a single feature to predict a continuous target variable.
- It inherits from the `IRegression` interface and provides implementation for the `fit`, `predict`, and `score` methods.
- Feature engineering can be applied to the input and output data using the `x_scalar` and `y_scalar` parameters.
- Private methods `_y_hat`, `_cost`, `_gradient` and `_validate_data` are used internally for different aspects of the linear regression algorithm.
