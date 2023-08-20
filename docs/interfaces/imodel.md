# Class [`IModel`](/learnML/interfaces/imodel.py#L5)

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

### [`fit(X: np.ndarray, Y: np.ndarray) -> None`](/learnML/interfaces/imodel.py#L9)

Fit the model to the provided data.

- `X` (np.ndarray): The input array-like object containing the data with shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The output array-like object containing the target data with shape `(n_samples, n_targets)` or `(n_samples,)`.

This method is responsible for training the model on the given data to learn patterns and relationships.

### [`predict(X: np.ndarray) -> np.ndarray`](/learnML/interfaces/imodel.py#L29)

Predict outputs based on the input data.

- `X` (np.ndarray): The input array-like object containing the data with shape `(n_samples, n_features)`.

This method utilizes the trained model to predict outputs corresponding to the input data.

### [`score(X: np.ndarray, Y: np.ndarray) -> float`](/learnML/interfaces/imodel.py#L48)

Calculate the score of the model's performance.

- `X` (np.ndarray): The input array-like object containing the data with shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The output array-like object containing the target data with shape `(n_samples, n_targets)` or `(n_samples,)`.

This method evaluates the model's performance by comparing its predictions with the actual target data and returns a score reflecting its accuracy.

## Notes

- The `IModel` interface serves as a blueprint for machine learning model classes, ensuring a consistent structure across different models.
- Model classes that implement this interface are expected to provide implementations for the `fit`, `predict`, and `score` methods.
- The methods' parameters and return types are designed to facilitate seamless integration and interoperability across different model implementations.
- This interface encourages modularity and standardization in the development of machine learning models, promoting code reusability and maintainability.
