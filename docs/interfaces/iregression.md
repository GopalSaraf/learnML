# Class `IRegression`

## Description

An abstract interface class for regression model classes in the `learnML` library. This class extends the `IModel` abstract class and provides additional methods and attributes specifically for regression models.

## Usage

```python
from learnML.interfaces import IRegression

# Define a custom regression model class that inherits from IRegression
class CustomRegressionModel(IRegression):
    def fit(self, X, Y):
        # Implementation of the fit method
        pass

    def predict(self, X):
        # Implementation of the predict method
        pass

    def score(self, X, Y):
        # Implementation of the score method
        pass
```

## Constructor

### `__init__(learning_rate: np.float64, n_iterations: int, debug: bool = True, copy_x: bool = True) -> None`

Initialize the regression model.

- `learning_rate` (np.float64): The learning rate.
- `n_iterations` (int): The number of iterations.
- `debug` (bool, optional): Whether to print debug messages. Default is `True`.
- `copy_x` (bool, optional): Whether to copy the input array. Default is `True`.

## Methods

### `fit(X: np.ndarray, Y: np.ndarray) -> None`

Fit the regression model to the data.

- `X` (np.ndarray): The array-like object containing the input data of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The array-like object containing the output data of shape `(n_samples, n_targets)` or `(n_samples,)`.

### `predict(X: np.ndarray) -> np.ndarray`

Predict the output given the input.

- `X` (np.ndarray): The array-like object containing the input data of shape `(n_samples, n_features)`.
- Returns: An array-like object containing the output data of shape `(n_samples, n_targets)` or `(n_samples,)`.

### `score(X: np.ndarray, Y: np.ndarray) -> float`

Calculate the score of the regression model.

- `X` (np.ndarray): The array-like object containing the input data of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The array-like object containing the output data of shape `(n_samples, n_targets)` or `(n_samples,)`.
- Returns: The score of the model as a float.

### `get_cost_history() -> np.ndarray`

Return the history of the cost function.

### `get_parameter_history() -> np.ndarray`

Return the history of the parameters.

### `get_weights() -> np.ndarray`

Return the weights.

### `get_intercept() -> np.float64`

Return the intercept.

### `_debug_print(iteration: int, cost: np.float64) -> None`

Print the current iteration and cost.

## Examples

```python
# Define a custom regression model class that inherits from IRegression
class CustomRegressionModel(IRegression):
    def fit(self, X, Y):
        # Implementation of the fit method
        pass

    def predict(self, X):
        # Implementation of the predict method
        pass

    def score(self, X, Y):
        # Implementation of the score method
        pass

# Create an instance of the custom regression model
regression_model = CustomRegressionModel(learning_rate=0.01, n_iterations=100)

# Fit the model to data
regression_model.fit(training_X, training_Y)

# Make predictions using the model
predictions = regression_model.predict(test_X)

# Calculate the model's score
model_score = regression_model.score(test_X, test_Y)

# Access cost history and parameters history
cost_history = regression_model.get_cost_history()
params_history = regression_model.get_parameter_history()

# Access model weights and intercept
weights = regression_model.get_weights()
intercept = regression_model.get_intercept()
```

## Notes

- The `IRegression` class builds upon the `IModel` interface and provides additional functionality tailored to regression models.
- Any regression model class that inherits from `IRegression` must implement the `fit`, `predict`, and `score` methods, as well as any other specific methods required.
