# Interface [`IRegression`](/learnML/interfaces/iregression.py#L7)

## Introduction

`IRegression` is an abstract interface that extends the `IModel` interface and specifies the methods and attributes expected from regression model classes. Regression models are used to predict continuous target variables based on input features.

## Usage

```python
from learnML.interfaces import IRegression

# Create a custom regression model class that implements the IRegression interface
class CustomRegression(IRegression):
    def __init__(self, learning_rate, n_iterations):
        # Initialize necessary attributes or hyperparameters
        super().__init__(learning_rate, n_iterations)

    def fit(self, X, Y):
        # Implement the training logic for the regression model
        pass

    def predict(self, X):
        # Implement the prediction logic for the regression model
        pass

    def score(self, X, Y):
        # Implement the scoring logic for the regression model
        pass

# Create an instance of the custom regression model
model = CustomRegression(learning_rate=0.01, n_iterations=1000)

# Load training data
X_train = ...
Y_train = ...

# Fit the model to the training data
model.fit(X_train, Y_train)

# Load test data
X_test = ...
Y_test = ...

# Make predictions using the trained model
predictions = model.predict(X_test)

# Evaluate the model's performance
score = model.score(X_test, Y_test)

# Get cost history and model parameters history
cost_history = model.get_cost_history()
params_history = model.get_parameter_history()

# Get learned weights and intercept
weights = model.get_weights()
intercept = model.get_intercept()
```

## Methods and Attributes

### Constructor

#### [`__init__(learning_rate: np.float64, n_iterations: int, debug: bool = True, copy_x: bool = True) -> None`](/learnML/interfaces/iregression.py#L10)

Initialize the regression model.

- `learning_rate` (np.float64): The learning rate, controlling the step size of optimization.
- `n_iterations` (int): The number of iterations for optimization.
- `debug` (bool, optional): Whether to print debug messages, by default True.
- `copy_x` (bool, optional): Whether to copy the input array, by default True.

This constructor sets up basic parameters for the regression model and initializes necessary attributes.

### Methods

#### [`fit(X: np.ndarray, Y: np.ndarray) -> None`](/learnML/interfaces/iregression.py#L44)

Fit the regression model to the provided data.

- `X` (np.ndarray): The input array-like object containing the data with shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The output array-like object containing the target data with shape `(n_samples, n_targets)` or `(n_samples,)`.

This method trains the regression model on the given data to learn relationships between input features and target variables.

#### [`predict(X: np.ndarray) -> np.ndarray`](/learnML/interfaces/iregression.py#L48)

Predict outputs based on the input data.

- `X` (np.ndarray): The input array-like object containing the data with shape `(n_samples, n_features)`.

This method utilizes the trained regression model to predict outputs corresponding to the input data.

#### [`score(X: np.ndarray, Y: np.ndarray) -> float`](/learnML/interfaces/iregression.py#L52)

Calculate the score of the regression model's performance.

- `X` (np.ndarray): The input array-like object containing the data with shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The output array-like object containing the target data with shape `(n_samples, n_targets)` or `(n_samples, )`.

This method evaluates the regression model's performance by comparing its predictions with the actual target data and returns a score reflecting its accuracy.

#### [`get_cost_history() -> np.ndarray`](/learnML/interfaces/iregression.py#L55)

Return the history of the cost function.

Returns an array containing the history of the cost function during training.

#### [`get_parameter_history() -> np.ndarray`](/learnML/interfaces/iregression.py#L66)

Return the history of the model parameters.

Returns an array containing the history of the model's parameters during training.

#### [`get_weights() -> np.ndarray`](/learnML/interfaces/iregression.py#L77)

Return the learned weights of the regression model.

Returns an array containing the learned weights.

#### [`get_intercept() -> np.float64`](/learnML/interfaces/iregression.py#L88)

Return the learned intercept of the regression model.

Returns the learned intercept value.

### Private Methods

#### [`_debug_print(iteration: int, cost: np.float64) -> None`](/learnML/interfaces/iregression.py#L99)

Print the current iteration and cost during training.

- `iteration` (int): The current iteration number.
- `cost` (np.float64): The current cost value.

## Notes

- The `IRegression` interface builds upon the `IModel` interface and provides additional methods and attributes specific to regression models.
- Model classes that implement this interface are expected to provide implementations for the `fit`, `predict`, and `score` methods, in addition to the methods inherited from `IModel`.
- The `get_cost_history`, `get_parameter_history`, `get_weights`, and `get_intercept` methods provide insights into the training process and learned model parameters.
- This interface promotes consistent structure and functionality across various regression model implementations.
