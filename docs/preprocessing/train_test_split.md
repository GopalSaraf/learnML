# Data Splitting Utilities

In machine learning, it's important to split your dataset into training and testing sets to evaluate the performance of your models. Here are some utility functions and classes that facilitate data splitting for various cross-validation techniques.

## Function [`_get_numpy_data(data: Union[pd.DataFrame, np.ndarray, list]) -> np.ndarray`](/learnML/preprocessing/train_test_split.py#L6)

Convert the input data to a numpy array.

- `data` (Union[pd.DataFrame, np.ndarray, list]): The data to be converted.

Returns:

- `np.ndarray`: The data as a numpy array.

## Function [`train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`](/learnML/preprocessing/train_test_split.py#L32)

Split the data into training and testing sets using a fixed proportion.

- `X` (np.ndarray): The input features of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): The output features of shape `(n_samples, n_outputs)`.
- `test_size` (float, optional): The size of the testing set, by default 0.2.
- `random_state` (int, optional): The random state, by default 0.

Returns:

- `tuple`: The training and testing sets `(X_train, X_test, Y_train, Y_test)`.

## Class [`KFoldSplit`](/learnML/preprocessing/train_test_split.py#L94)

Class for splitting the data into training and testing sets using k-fold cross-validation.

### Constructor [`KFoldSplit(X: np.ndarray, Y: np.ndarray = None, k: int = 5)`](/learnML/preprocessing/train_test_split.py#L97)

Initialize the `KFoldSplit` class.

- `X` (np.ndarray): The input features.
- `Y` (np.ndarray, optional): The output features, by default None.
- `k` (int, optional): The number of folds, by default 5.

### Method [`split() -> None`](/learnML/preprocessing/train_test_split.py#L121)

Split the data into training and testing sets using k-fold cross-validation.

### Method [`get_fold(nthFold: int = 0) -> tuple`](/learnML/preprocessing/train_test_split.py#L149)

Get the training and testing sets for the nth fold.

- `nthFold` (int, optional): The fold number, by default 0.

Returns:

- `tuple`: The training and testing sets `(X_train, X_test, Y_train, Y_test)` if Y is not None, otherwise `(X_train, X_test)`.

## Class [`OneLeaveOutSplit`](/learnML/preprocessing/train_test_split.py#L193)

Class for splitting the data into training and testing sets using leave-one-out cross-validation.

### Constructor [`OneLeaveOutSplit(X: np.ndarray, Y: np.ndarray = None)`](/learnML/preprocessing/train_test_split.py#L196)

Initialize the `OneLeaveOutSplit` class.

- `X` (np.ndarray): The input features.
- `Y` (np.ndarray, optional): The output features, by default None.

### Method [`randomize() -> None`](/learnML/preprocessing/train_test_split.py#L216)

Randomize the data.

### Method [`get_fold(nthFold: int = 0) -> tuple`](/learnML/preprocessing/train_test_split.py#L237)

Get the training and testing sets for the nth fold.

- `nthFold` (int, optional): The fold number, by default 0.

Returns:

- `tuple`: The training and testing sets `(X_train, X_test, Y_train, Y_test)` if Y is not None, otherwise `(X_train, X_test)`.

## Notes

- These utilities assist in splitting data for training and evaluating machine learning models using different cross-validation techniques.
- The `train_test_split` function provides a simple way to perform a single train-test split.
- The `KFoldSplit` class allows k-fold cross-validation where the data is divided into k subsets, and each fold is used as a testing set while the remaining data is used for training.
- The `OneLeaveOutSplit` class implements leave-one-out cross-validation, where each data point is used as a testing sample while the rest are used for training.
