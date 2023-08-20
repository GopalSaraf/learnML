# Class `IFeatureEngineering`

## Description

An abstract interface class for feature engineering classes in the `learnML` library. This class outlines the methods that any feature engineering class in the library should implement. Feature engineering classes derived from `IFeatureEngineering` are designed to preprocess and transform input data for machine learning tasks.

## Usage

```python
from learnML.interfaces import IFeatureEngineering

# Define a custom feature engineering class that inherits from IFeatureEngineering
class CustomFeatureEngineering(IFeatureEngineering):
    def fit(self, data=None):
        # Implementation of the fit method
        pass

    def transform(self, data=None):
        # Implementation of the transform method
        pass

    def fit_transform(self, data=None):
        # Implementation of the fit_transform method
        pass

    def inverse_transform(self, data):
        # Implementation of the inverse_transform method
        pass
```

## Constructor

### `__init__(data: np.ndarray) -> None`

Initialize the feature engineering instance with input data.

- `data` (np.ndarray): The input array of shape `(n_samples, n_features)`.

## Methods

### `fit(data: np.ndarray = None) -> None`

Fit the feature engineering instance to data.

- `data` (np.ndarray, optional): The input array of shape `(n_samples, n_features)`. Default is `None` (uses the input array passed in the constructor).

### `transform(data: np.ndarray = None) -> np.ndarray`

Transform data using the feature engineering instance.

- `data` (np.ndarray, optional): The input array of shape `(n_samples, n_features)`. Default is `None` (uses the input array passed in the constructor).
- Returns: The transformed data of shape `(n_samples, n_features)`.

### `fit_transform(data: np.ndarray = None) -> np.ndarray`

Fit the feature engineering instance with data and transform data using it.

- `data` (np.ndarray, optional): The input array of shape `(n_samples, n_features)`. Default is `None` (uses the input array passed in the constructor).
- Returns: The transformed data of shape `(n_samples, n_features)`.

### `inverse_transform(data: np.ndarray) -> np.ndarray`

Convert the data back to the original representation.

- `data` (np.ndarray): The input array of shape `(n_samples, n_features)`.
- Returns: The transformed data of shape `(n_samples, n_features)`.

## Examples

```python
# Define a custom feature engineering class that inherits from IFeatureEngineering
class CustomFeatureEngineering(IFeatureEngineering):
    def fit(self, data=None):
        # Implementation of the fit method
        pass

    def transform(self, data=None):
        # Implementation of the transform method
        pass

    def fit_transform(self, data=None):
        # Implementation of the fit_transform method
        pass

    def inverse_transform(self, data):
        # Implementation of the inverse_transform method
        pass

# Create an instance of the custom feature engineering class
feature_engineering = CustomFeatureEngineering(data=input_data)

# Fit the feature engineering instance to data
feature_engineering.fit()

# Transform input data using the feature engineering instance
transformed_data = feature_engineering.transform()

# Inverse transform the transformed data
original_data = feature_engineering.inverse_transform(transformed_data)
```

## Notes

- The `IFeatureEngineering` class provides a common interface for all feature engineering classes in the `learnML` library.
- Any feature engineering class that inherits from `IFeatureEngineering` must implement the `fit`, `transform`, `fit_transform`, and `inverse_transform` methods.
