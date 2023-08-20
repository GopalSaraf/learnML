# Interface [`IFeatureEngineering`](/learnML/interfaces/ifeature_engineering.py#L5)

## Introduction

`IFeatureEngineering` is an abstract interface that defines methods for performing feature engineering on input data. Feature engineering involves transforming and preprocessing raw input data to create more informative and suitable features for machine learning models.

## Usage

```python
from learnML.interfaces import IFeatureEngineering

# Create a custom feature engineering class that implements the IFeatureEngineering interface
class CustomFeatureEngineering(IFeatureEngineering):
    def __init__(self, data):
        # Initialize necessary attributes or parameters
        super().__init__(data)

    def fit(self, data=None):
        # Implement the fitting logic for the feature engineering
        pass

    def transform(self, data=None):
        # Implement the transformation logic for the feature engineering
        pass

    def fit_transform(self, data=None):
        # Implement the fitting and transformation logic for the feature engineering
        pass

    def inverse_transform(self, data):
        # Implement the inverse transformation logic for the feature engineering
        pass

# Create an instance of the custom feature engineering class
feature_engineer = CustomFeatureEngineering(data=...)

# Fit the feature engineering object to the data
feature_engineer.fit()

# Transform the data using the feature engineering
transformed_data = feature_engineer.transform()

# Inverse transform the transformed data back to the original representation
original_data = feature_engineer.inverse_transform(transformed_data)
```

## Methods

### Constructor

#### [`__init__(data: np.ndarray) -> None`](/learnML/interfaces/ifeature_engineering.py#L9)

Initialize the feature engineering object.

- `data` (np.ndarray): The input data array with shape `(n_samples, n_features)`.

This constructor sets up the input data for feature engineering.

### Methods

#### [`fit(data: np.ndarray = None) -> None`](/learnML/interfaces/ifeature_engineering.py#L20)

Fit the feature engineering object to the data.

- `data` (np.ndarray, optional): The input data array with shape `(n_samples, n_features)`. If not provided, the input data passed to the constructor will be used.

This method adapts the feature engineering object to the input data if necessary.

#### [`transform(data: np.ndarray = None) -> np.ndarray`](/learnML/interfaces/ifeature_engineering.py#L33)

Transform data using the feature engineering object.

- `data` (np.ndarray, optional): The input data array with shape `(n_samples, n_features)`. If not provided, the input data passed to the constructor will be used.

Returns the transformed data array with shape `(n_samples, n_features)`.

#### [`fit_transform(data: np.ndarray = None) -> np.ndarray`](/learnML/interfaces/ifeature_engineering.py#L51)

Fit the feature engineering object with data and transform it.

- `data` (np.ndarray, optional): The input data array with shape `(n_samples, n_features)`. If not provided, the input data passed to the constructor will be used.

Returns the transformed data array with shape `(n_samples, n_features)`.

#### [`inverse_transform(data: np.ndarray) -> np.ndarray`](/learnML/interfaces/ifeature_engineering.py#L69)

Convert the transformed data back to the original representation.

- `data` (np.ndarray): The transformed data array with shape `(n_samples, n_features)`.

Returns the data array with shape `(n_samples, n_features)` in the original representation.

## Notes

- The `IFeatureEngineering` interface provides a structured approach to implementing feature engineering classes that can be used with various machine learning models.
- Feature engineering can include tasks such as scaling, normalization, one-hot encoding, and more.
- Model classes that require feature engineering can accept instances of classes implementing this interface to preprocess and transform data effectively.
