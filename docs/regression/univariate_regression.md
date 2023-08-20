# Class [`UnivariateLinearRegression`](/learnML/regression/univariate_regression.py#L7)

## Introduction

Linear regression is a fundamental supervised machine learning algorithm that models the relationship between a dependent variable and a single independent variable. It approximates this relationship using a linear equation. Univariate Linear Regression is particularly useful when there is a clear linear correlation between the input and output variables.

## Mathematical Approach

Univariate Linear Regression aims to find the best-fitting line that predicts the output variable based on the input feature. The linear equation is represented as:

```
y = mx + b
```

Where:

- `y` is the predicted output (target variable).
- `x` is the input feature (independent variable).
- `m` is the slope of the line (weight).
- `b` is the y-intercept.

The goal is to determine the optimal values of `m` and `b` that minimize the difference between predicted values and actual target values.

## Usage

To utilize the Univariate Linear Regression model, follow these steps:

1. Import the `UnivariateLinearRegression` class from the appropriate module.
2. Create an instance of the `UnivariateLinearRegression` class, specifying hyperparameters.
3. Fit the model to your training data using the `fit` method.
4. Make predictions on new data using the `predict` method.
5. Evaluate the model's performance using the `score` method.

```python
from learnML.regression import UnivariateLinearRegression

# Create an instance of UnivariateLinearRegression
model = UnivariateLinearRegression(learning_rate=0.01, n_iterations=1000)

# Fit the model to training data
model.fit(X_train, Y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Calculate the model's score
model_score = model.score(X_test, Y_test)
```

## Implementation

### Constructor

#### [`__init__(learning_rate: np.float64 = 0.001, n_iterations: int = 1000, x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, y_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, debug: bool = True, copy_x: bool = True) -> None`](/learnML/regression/univariate_regression.py#L24)

Initialize the Univariate Linear Regression model.

- `learning_rate` (np.float64, optional): The learning rate controls the step size during optimization.
- `n_iterations` (int, optional): The number of iterations for optimization.
- `x_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for input data.
- `y_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for output data.
- `debug` (bool, optional): Whether to print debug messages during training.
- `copy_x` (bool, optional): Whether to make a copy of the input array to avoid modifying the original data.

### Methods

#### [`fit(X: np.ndarray, Y: np.ndarray, w: np.float64 = 0.0, b: np.float64 = 0.0) -> None`](/learnML/regression/univariate_regression.py#L206)

Train the Univariate Linear Regression model.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray): The output array of shape `(n_samples,)`.
- `w` (np.float64, optional): Initial weight.
- `b` (np.float64, optional): Initial intercept.

**Mathematical Explanation:**

The `fit` method uses **Gradient Descent** to adjust the weight (`w`) and intercept (`b`) iteratively. The parameter updates are given by:

```
new_weight = old_weight - learning_rate * dw
new_intercept = old_intercept - learning_rate * db
```

Where `dw` is the gradient of the cost function with respect to the weight, and `db` is the gradient with respect to the intercept.

The `_gradient` method calculates these gradients.

#### [`predict(X: Union[np.ndarray, np.float64]) -> Union[np.ndarray, np.float64]`](/learnML/regression/univariate_regression.py#L269)

Predict the output using the trained model.

- `X` (Union[np.ndarray, np.float64]): The input value or array of shape `(n_samples,)`.

**Mathematical Explanation:**

The predicted output `y_hat` is calculated using the linear equation:

```
y_hat = mx + b
```

Where:

- `y_hat` is the predicted output.
- `x` is the input value or array.
- `m` is the learned weight.
- `b` is the learned intercept.

#### [`score(X: np.ndarray, Y: np.ndarray, w: np.float64 = None, b: np.float64 = None) -> np.float64`](/learnML/regression/univariate_regression.py#L310)

Calculate the cost function given input and output data.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray): The output array of shape `(n_samples,)`.
- `w` (np.float64, optional): The weight.
- `b` (np.float64, optional): The intercept.

**Mathematical Explanation:**

The cost function (Mean Squared Error) is calculated as:

```
MSE = (1 / (2 * m)) * Σ(y_hat_i - y_i)^2
```

Where:

- `m` is the number of samples.
- `y_hat_i` is the predicted value for the `i`-th example.
- `y_i` is the actual target value for the `i`-th example.

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

**Mathematical Explanation:**

The `_gradient` method computes the gradients of the cost function with respect to the weights and intercept. The gradients are used for parameter updates during optimization. The gradient of the cost function is calculated using the partial derivatives:

```
dw = (1 / m) * Σ(y_hat_i - y_i) * x_i
db = (1 / m) * Σ(y_hat_i - y_i)
```

Where:

- `m` is the number of samples.
- `y_hat_i` is the predicted value for the `i`-th example.
- `y_i` is the actual target value for the `i`-th example.
- `x_i` is the `i`-th feature of the input data.

#### [`_validate_data(X: np.ndarray, Y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]`](/learnML/regression/univariate_regression.py#L156)

Validate and preprocess input and output data.

- `X` (np.ndarray): The input array of shape `(n_samples,)`.
- `Y` (np.ndarray or None, optional): The target array of shape `(n_samples,)`.

The `_validate_data` method preprocesses input and output data, applying feature engineering if provided, and ensures data consistency and compatibility.

## Notes

- The `UnivariateLinearRegression` class performs linear regression on a single feature to predict a continuous target variable.
- It inherits from the `IRegression` interface and provides implementation for the `fit`, `predict`, and `score` methods.
- Feature engineering can be applied to the input and output data using the `x_scalar` and `y_scalar` parameters.
- Private methods `_y_hat`, `_cost`, `_gradient` and `_validate_data` are used internally for different aspects of the linear regression algorithm.
