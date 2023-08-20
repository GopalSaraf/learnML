# Z-Score Normalization (Standardization) Class

This class provides Z-score normalization (also known as standardization) for input data. Z-score normalization transforms data so that it has a mean of 0 and a standard deviation of 1. This is achieved by subtracting the mean of each column and dividing by the standard deviation of each column.

## Mathematical Explanation

Z-score normalization (standardization) is a statistical technique used to transform features in a dataset to have a mean of 0 and a standard deviation of 1. This transformation is applied column-wise to the features.

For a given feature `X` with values `x_1, x_2, ..., x_n`, the Z-score normalization is calculated as follows:

- Calculate the mean (`μ`) of the feature:

```
μ = (x_1 + x_2 + ... + x_n) / n
```

- Calculate the standard deviation (`σ`) of the feature:

```
σ = sqrt(((x_1 - μ)^2 + (x_2 - μ)^2 + ... + (x_n - μ)^2) / n)
```

- For each value `x_i` in the feature, compute the Z-score:

```
z_i = (x_i - μ) / σ
```

The Z-score normalization ensures that the transformed values have a mean of 0 and a standard deviation of 1, which is beneficial for many machine learning algorithms that assume standardized data. It also helps in comparing and visualizing features on a similar scale, avoiding potential issues caused by features with different scales.

The `ZScoreNormalization` class provides an implementation of this mathematical process, allowing you to easily normalize your data using Z-score normalization.

## Class [`ZScoreNormalization`](/learnML/preprocessing/z_score_normalization.py#L7)

Z-score normalization (standardization) for input data.

### Constructor [`ZScoreNormalization(data: np.ndarray, index: Union[int, list, range] = None)`](/learnML/preprocessing/z_score_normalization.py#L10)

Initialize the `ZScoreNormalization` class.

- `data` (np.ndarray): The data to be normalized.
- `index` (Union[int, list, range], optional): The index of the columns to be normalized.

### Method [`_get_mean() -> np.ndarray`](/learnML/preprocessing/z_score_normalization.py#L33)

Calculate the mean of each column.

Returns:

- `np.ndarray`: The mean of each column.

### Method [`_get_std() -> np.ndarray`](/learnML/preprocessing/z_score_normalization.py#L46)

Calculate the standard deviation of each column.

Returns:

- `np.ndarray`: The standard deviation of each column.

### Method [`fit(data: np.ndarray = None) -> None`](/learnML/preprocessing/z_score_normalization.py#L59)

Fit the data to calculate mean and standard deviation.

- `data` (np.ndarray, optional): The data to be normalized, by default None.

### Method [`transform(data: np.ndarray) -> np.ndarray`](/learnML/preprocessing/z_score_normalization.py#L74)

Transform the input data using Z-score normalization.

- `data` (np.ndarray): The data to be normalized.

Returns:

- `np.ndarray`: The normalized data.

### Method [`fit_transform(data: np.ndarray = None) -> np.ndarray`](/learnML/preprocessing/z_score_normalization.py#L100)

Fit the data and transform it using Z-score normalization.

- `data` (np.ndarray, optional): The data to be normalized, by default None.

Returns:

- `np.ndarray`: The normalized data.

### Method [`inverse_transform(data: np.ndarray) -> np.ndarray`](/learnML/preprocessing/z_score_normalization.py#L119)

Inverse the normalization and transform the data back to the original representation.

- `data` (np.ndarray): The data to be inverse-transformed.

Returns:

- `np.ndarray`: The inverse-transformed data.
