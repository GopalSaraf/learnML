# Class `IModel`

## Description

The `IModel` abstract class serves as a foundational interface for all model classes within the `learnML` library. By providing a consistent structure, it ensures that each model adheres to a common set of methods and behaviors. Models that inherit from `IModel` are expected to implement the `fit`, `predict`, and `score` methods, allowing users to seamlessly interchange and evaluate different models in their machine learning pipelines.

## Usage

```python
from learnML.interfaces import IModel

# Define a custom model class that inherits from IModel
class CustomModel(IModel):
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

## Methods

### `fit(X: np.ndarray, Y: np.ndarray) -> None`

Fit the model to the data.

- `X` (np.ndarray): The array-like object containing the input data of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The array-like object containing the output data of shape `(n_samples, n_targets)` or `(n_samples,)`.

### `predict(X: np.ndarray) -> np.ndarray`

Predict the output given the input.

- `X` (np.ndarray): The array-like object containing the input data of shape `(n_samples, n_features)`.
- Returns: An array-like object containing the output data of shape `(n_samples, n_targets)` or `(n_samples,)`.

### `score(X: np.ndarray, Y: np.ndarray) -> float`

Calculate the score of the model.

- `X` (np.ndarray): The array-like object containing the input data of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The array-like object containing the output data of shape `(n_samples, n_targets)` or `(n_samples,)`.
- Returns: The score of the model as a float.

## Examples

```python
# Define a custom model class that inherits from IModel
class CustomModel(IModel):
    def fit(self, X, Y):
        # Implementation of the fit method
        pass

    def predict(self, X):
        # Implementation of the predict method
        pass

    def score(self, X, Y):
        # Implementation of the score method
        pass

# Create an instance of the custom model
model = CustomModel()

# Fit the model to data
model.fit(training_X, training_Y)

# Make predictions using the model
predictions = model.predict(test_X)

# Calculate the model's score
model_score = model.score(test_X, test_Y)
```

## Notes

- The `IModel` class provides a common interface for all model classes in the `learnML` library.
- Any model class that inherits from `IModel` must implement the `fit`, `predict`, and `score` methods.
