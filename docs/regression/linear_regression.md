# Class [`LinearRegression`](/learnML/regression/linear_regression.py#L7)

## Introduction

Linear Regression is a fundamental supervised machine learning algorithm used to model the relationship between one or more independent variables and a continuous target variable. It serves as a cornerstone for predictive modeling and provides insights into how changes in input features affect the target variable. Linear Regression assumes that the target variable can be expressed as a linear combination of the input features, facilitating interpretation and prediction.

## Mathematical Approach

Linear Regression aims to find the best-fitting linear equation that predicts the target variable. The equation takes the form:

```
y = b + w1 * x1 + w2 * x2 + ... + wn * xn
```

Where:

- `y` is the predicted target variable.
- `b` is the intercept (bias term).
- `w1, w2, ..., wn` are the weights assigned to each input feature `x1, x2, ..., xn`.

The goal of Linear Regression is to find the optimal values for `b` and the weights that minimize the difference between predicted and actual target values.

## Usage

To use the Linear Regression model, follow these steps:

1. Import the `LinearRegression` class from the appropriate module.
2. Create an instance of the `LinearRegression` class, specifying hyperparameters.
3. Fit the model to your training data using the `fit` method.
4. Make predictions on new data using the `predict` method.
5. Evaluate the model's performance using the `score` method.

```python
from learnML.regression import LinearRegression

# Create an instance of LinearRegression
model = LinearRegression(learning_rate=0.01, n_iterations=1000)

# Fit the model to training data
model.fit(X_train, Y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Calculate the model's score
model_score = model.score(X_test, Y_test)
```

## Implementation

### Constructor

#### [`__init__(learning_rate: np.float64 = 0.001, n_iterations: int = 1000, lambda_: np.float64 = 0, x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, y_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, debug: bool = True, copy_x: bool = True) -> None`](/learnML/regression/linear_regression.py#L24)

Initialize the Linear Regression model.

- `learning_rate` (np.float64, optional): The learning rate controls the step size during optimization.
- `n_iterations` (int, optional): The number of iterations for optimization.
- `lambda_` (np.float64, optional): The regularization parameter for L2 regularization.
- `x_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for input data.
- `y_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for output data.
- `debug` (bool, optional): Whether to print debug messages.
- `copy_x` (bool, optional): Whether to copy the input array to avoid modification.

### Methods

#### [`fit(X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0) -> None`](/learnML/regression/linear_regression.py#L214)

Train the Linear Regression model.

- `X` (np.ndarray): The input array of shape `(n_samples, n_features)` or `(n_samples,)`.
- `Y` (np.ndarray): The output array of shape `(n_samples,)` or `(n_samples, 1)`.
- `W` (np.ndarray, optional): Initial weight array.
- `b` (np.float64, optional): Initial intercept.

**Mathematical Explanation:**

The training process uses the **Gradient Descent** algorithm to minimize the Mean Squared Error (MSE) cost function. The weights (`W`) and intercept (`b`) are iteratively updated using the following formulas:

```
new_weight = old_weight - learning_rate * dw
new_intercept = old_intercept - learning_rate * db
```

Where `dw` is the gradient of the cost function with respect to weights and `db` is the gradient with respect to the intercept.

To compute `dw` and `db`, the `_gradient` method is utilized.

#### [`predict(X: np.ndarray) -> np.ndarray`](/learnML/regression/linear_regression.py#L275)

Predict the target variable using the trained model.

- `X` (np.ndarray): The input array of shape `(n_samples, n_features)` or `(n_samples,)`.

**Mathematical Explanation:**

The predicted target variable `y_hat` is calculated using the linear equation:

```
y_hat = b + w1 * x1 + w2 * x2 + ... + wn * xn
```

Where:

- `y_hat` is the predicted target variable.
- `b` is the intercept.
- `w1, w2, ..., wn` are the learned weights.
- `x1, x2, ..., xn` are the input features.

#### [`score(X: np.ndarray, Y: np.ndarray, w: np.ndarray = None, b: np.float64 = None) -> np.float64`](/learnML/regression/linear_regression.py#L305)

Calculate the cost function (Mean Squared Error) given input and target data.

- `X` (np.ndarray): The input array of shape `(n_samples, n_features)` or `(n_samples,)`.
- `Y` (np.ndarray): The target array of shape `(n_samples,)` or `(n_samples, 1)`.
- `w` (np.ndarray, optional): Weight array.
- `b` (np.float64, optional): Intercept.

**Mathematical Explanation:**

The cost function (Mean Squared Error) measures the average squared difference between predicted and actual target values:

```
MSE = (1 / (2 * m)) * Σ(y_hat_i - y_i)^2
```

Where:

- `m` is the number of samples.
- `y_hat_i` is the predicted value for the `i`-th example.
- `y_i` is the actual target value for the `i`-th example.

### Private Methods

#### [`_y_hat(X: np.ndarray, W: np.ndarray, b: np.float64) -> np.float64`](/learnML/regression/linear_regression.py#L73)

Calculate the predicted value of the target variable given input features, weights, and intercept.

- `X` (np.ndarray): The input array of shape `(n_features,)`.
- `W` (np.ndarray): The weight array of shape `(n_features,)`.
- `b` (np.float64): The intercept.

**Mathematical Explanation:**

The `_y_hat` method computes the predicted target value based on the linear equation:

```
y_hat = b + w1 * x1 + w2 * x2 + ... + wn * xn
```

#### [`_cost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.float64) -> np.float64`](/learnML/regression/linear_regression.py#L93)

Calculate the cost function

(Mean Squared Error) given input `X`, target `Y`, weights `W`, and intercept `b`.

- `X` (np.ndarray): The input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The target array of shape `(n_samples,)`.
- `W` (np.ndarray): The weight array of shape `(n_features,)`.
- `b` (np.float64): The intercept.

**Mathematical Explanation:**

The `_cost` method computes the MSE cost function based on the predicted target values and actual target values. It uses the formula:

```
MSE = (1 / (2 * m)) * Σ(y_hat_i - y_i)^2
```

Where:

- `m` is the number of samples.
- `y_hat_i` is the predicted value for the `i`-th example.
- `y_i` is the actual target value for the `i`-th example.

#### [`_gradient(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.float64) -> Tuple[np.ndarray, np.float64]`](/learnML/regression/linear_regression.py#L129)

Calculate the gradient of the cost function with respect to weights and intercept.

- `X` (np.ndarray): The input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The target array of shape `(n_samples,)`.
- `W` (np.ndarray): The weight array of shape `(n_features,)`.
- `b` (np.float64): The intercept.

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

#### [`_validate_data(X: np.ndarray, Y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]`](/learnML/regression/linear_regression.py#L171)

Validate and preprocess input and output data.

- `X` (np.ndarray): The input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray, optional): The target array of shape `(n_samples,)` or `(n_samples, 1)`.

The `_validate_data` method preprocesses input and output data, applying feature engineering if provided, and ensures data consistency and compatibility.

## Notes

- The `LinearRegression` class implements linear regression for predicting a continuous target variable using one or more input features.
- It inherits from the `IRegression` interface and provides implementations for the `fit`, `predict`, and `score` methods.
- Feature engineering can be applied to the input and output data using the `x_scalar` and `y_scalar` parameters.
- Private methods `_y_hat`, `_cost`, `_gradient`, and `_validate_data` are used internally for different aspects of the linear regression algorithm.
