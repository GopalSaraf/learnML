# Polynomial Feature Generator Class

The `PolynomialFeatures` class is designed to generate polynomial features from input data. Polynomial features are derived from the original features by raising them to various degrees. This process can help capture non-linear relationships between features and improve the performance of machine learning algorithms.

## Usage

```python
import numpy as np
from polynomial_features import PolynomialFeatures

# Sample input data
data = np.array([[1, 2], [3, 4], [5, 6]])

# Initialize PolynomialFeatures with degree 2
poly_features = PolynomialFeatures(data, degree=2)

# Generate polynomial features
polynomial_data = poly_features.transform()

print(polynomial_data)
```

This will generate polynomial features of degree 1 and 2 for all features in the input data.

The `PolynomialFeatures` class allows you to easily generate polynomial features from input data, which can be useful for enhancing the predictive power of machine learning models by capturing complex relationships between features.

## Class [`PolynomialFeatures`](/learnML/preprocessing/polynomial_features.py#L5)

## Constructor

### [`PolynomialFeatures(data: np.ndarray, degree: Union[int, List[int], Dict[int, Union[int, List[int]]]] = 2) -> None`](/learnML/preprocessing/polynomial_features.py#L8)

Initialize the `PolynomialFeatures` class.

- `data` (np.ndarray): The input array of shape `(n_samples, n_features)`.
- `degree` (Union[int, List[int], Dict[int, Union[int, List[int]]]], optional): The degree of the polynomial, by default 2.

  - It can be a single integer, a list of integers, or a dictionary of integers and lists of integers.
  - If it's a single integer, polynomial features of all features will be generated with degrees from 1 to the given integer.
  - If it's a list of integers, polynomial features of the features will be generated with the specified degrees in the list.
  - If it's a dictionary of integers and lists of integers, each feature's polynomial features will be generated according to the provided degrees.

  Examples:

  ```python
  degree = 2
  # Generate polynomial features of degree 1 and 2 for all features

  degree = [2, 3]
  # Generate polynomial features of degree 2 and 3 for all features

  degree = {0: [2, 3], 1: 2}
  # Generate polynomial features of degree 2 and 3 for the first feature
  # Generate polynomial features of degree 1 and 2 for the second feature
  ```

## Method

### [`_get_degree(data: np.float64, feature_idx: int, degree: int) -> np.ndarray`](/learnML/preprocessing/polynomial_features.py#L60)

Calculate the polynomial of a given degree for a specified feature.

- `data` (np.float64): The input value.
- `feature_idx` (int): The index of the feature.
- `degree` (int): The degree of the polynomial.

Returns:

- `np.ndarray`: The polynomial of the given degree for the specified feature.

### [`transform(data: np.ndarray = None) -> np.ndarray`](/learnML/preprocessing/polynomial_features.py#L81)

Generate polynomial features from input data.

- `data` (np.ndarray, optional): The input array of shape `(n_samples, n_features)`, by default None (uses the input array passed in the constructor).

Returns:

- `np.ndarray`: The polynomial features of the input array of shape `(n_samples, n_features)`.
