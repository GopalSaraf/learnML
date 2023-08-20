# Class [`LogisticRegression`](/learnML/classification/logistic_regression.py#L7)

## Introduction

Logistic Regression is a fundamental classification algorithm used to model the probability of a binary outcome. It's widely employed in machine learning for binary classification tasks and offers insights into the relationship between input features and class probabilities.

## Mathematical Approach

Logistic Regression aims to predict the probability of a binary outcome by modeling it as a sigmoid function of a linear combination of input features. The equation takes the form:

```
P(y=1 | X) = 1 / (1 + e^(-z))
```

Where `P(y=1 | X)` is the probability of the positive class given input `X`, and `z` is the linear combination of input features, weights, and an intercept.

## Usage

To use the Logistic Regression model, follow these steps:

1. Import the `LogisticRegression` class from the appropriate module.
2. Create an instance of the `LogisticRegression` class, specifying hyperparameters.
3. Fit the model to your training data using the `fit` method.
4. Make predictions on new data using the `predict` method.
5. Evaluate the model's performance using the `score` method.

```python
from learnML.classification import LogisticRegression

# Create an instance of LogisticRegression
model = LogisticRegression(learning_rate=0.001, n_iterations=1000)

# Fit the model to training data
model.fit(X_train, Y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Calculate the model's score
model_score = model.score(X_test, Y_test)
```

## Implementation

### Constructor

#### [`__init__(learning_rate: np.float64 = 0.001, n_iterations: int = 1000, lambda_: np.float64 = 0, x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, debug: bool = True, copy_x: bool = True) -> None`](/learnML/classification/logistic_regression.py#L26)

Initialize the Logistic Regression model.

- `learning_rate` (np.float64, optional): The learning rate controlling optimization step size.
- `n_iterations` (int, optional): Number of optimization iterations.
- `lambda_` (np.float64, optional): Regularization parameter.
- `x_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for input data.
- `debug` (bool, optional): Debug message printing.
- `copy_x` (bool, optional): Copy input array to avoid modification.

### Methods

#### [`fit(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0.0) -> None`](/learnML/classification/logistic_regression.py#L243)

Train the model.

**Mathematical Explanation:**

The training process involves iteratively updating the weights and intercept using gradient descent to minimize the cost function. The weights are updated using the formula:

```
new_weight = old_weight - learning_rate * dw
```

And the intercept is updated similarly. The gradient of the cost function is computed using the `_gradient` method.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): Output array of shape `(n_samples,)`.
- `W` (np.ndarray, optional): Weight array of shape `(n_features,)`.
- `b` (np.float64, optional): Intercept value.

#### [`predict_proba(self, X: np.ndarray) -> np.ndarray`](/learnML/classification/logistic_regression.py#L310)

Predict the probability of the output given input.

**Mathematical Explanation:**

The predicted probability

`P(y=1 | X)` is computed using the sigmoid function applied to the linear combination of input features, weights, and the intercept, as shown in the `_y_hat` method.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.

#### [`predict(self, X: np.ndarray) -> np.ndarray`](/learnML/classification/logistic_regression.py#L333)

Predict the output given input.

**Mathematical Explanation:**

The predicted output `y_pred` is obtained by rounding the predicted probabilities obtained from the `predict_proba` method.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.

#### [`score(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = None) -> np.float64`](/learnML/classification/logistic_regression.py#L351)

Compute the cost of the model given input `X`, output `Y`, weights `W`, and intercept `b`.

**Mathematical Explanation:**

The cost of the model is computed using the `_cost` method, as explained earlier.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): Output array of shape `(n_samples,)`.
- `W` (np.ndarray, optional): Weight array of shape `(n_features,)`.
- `b` (np.float64, optional): Intercept value.

### Private Methods

#### [`_sigmoid(self, z: np.float64) -> np.float64`](/learnML/classification/logistic_regression.py#L65)

Compute the sigmoid function.

**Mathematical Explanation:**

The sigmoid function is defined as:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Where `z` is the input parameter. It maps any real value to the range [0, 1], making it suitable for representing probabilities.

#### [`_y_hat(self, X: np.ndarray, W: np.ndarray, b: np.float64) -> Union[np.float64, np.ndarray]`](/learnML/classification/logistic_regression.py#L82)

Calculate predicted value of `y` given input `X`, weights `W`, and intercept `b`.

**Mathematical Explanation:**

The predicted value `y_hat` is computed using the sigmoid function applied to the linear combination of input features, weights, and the intercept:

```
y_hat = sigmoid(z) = 1 / (1 + e^(-z))
```

Where `z` is calculated as:

```
z = X * W + b
```

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `W` (np.ndarray): Weight array of shape `(n_features,)`.
- `b` (np.float64): Intercept value.

#### [`_cost(self, X: np.ndarray, y: np.float64, W: np.ndarray, b: np.float64) -> np.float64`](/learnML/classification/logistic_regression.py#L107)

Compute cost of the model given input `X`, output `y`, weights `W`, and intercept `b`.

**Mathematical Explanation:**

The cost of logistic regression is calculated using the cross-entropy loss formula:

```
cost = -[y * log(y_hat) + (1 - y) * log(1 - y_hat)]
```

Where `y` is the true output, and `y_hat` is the predicted output.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `y` (np.float64): Output array of shape `(n_samples,)`.
- `W` (np.ndarray): Weight array of shape `(n_features,)`.
- `b` (np.float64): Intercept value.

#### [`_gradient(self, X: np.ndarray, Y: np.float64, W: np.ndarray, b: np.float64) -> Union[np.ndarray, np.float64]`](/learnML/classification/logistic_regression.py#L152)

Compute gradient of the model with respect to weights and intercept.

**Mathematical Explanation:**

The gradient of the cost function with respect to weights `dw` and the intercept `db` is calculated using the formulas:

```
dw = (1 / m) * X.T * (y_hat - y)
db = (1 / m) * sum(y_hat - y)
```

Where `m` is the number of samples, `X.T` is the transpose of the input matrix, and `y_hat` is the predicted output.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `Y` (np.float64): Output array of shape `(n_samples,)`.
- `W` (np.ndarray): Weight array of shape `(n_features,)`.
- `b` (np.float64): Intercept value.

#### [`_validate_data(self, X: np.ndarray, Y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]`](/learnML/classification/logistic_regression.py#L206)

Validate and preprocess input and output data.

**Mathematical Explanation:**

This method validates input and output data, ensuring that they have the correct shapes and applying any specified feature engineering transformations.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray, optional): Output array of shape `(n_samples,)`.

## Notes

- The `LogisticRegression` class implements logistic regression for predicting a binary target variable using one or more input features.
- It inherits from the `IRegression` interface and provides implementations for the `fit`, `predict` and `score` methods.
- Feature engineering can be applied to the input and output data using the `x_scalar` and `y_scalar` parameters.
- Private methods `_y_hat`, `_cost`, `_gradient`, and `_validate_data` are used internally for different aspects of the logistic regression algorithm.
