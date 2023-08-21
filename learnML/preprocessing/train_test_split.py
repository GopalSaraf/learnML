import numpy as np
import pandas as pd
from typing import Tuple, Union


def _get_numpy_data(data: Union[pd.DataFrame, np.ndarray, list]) -> np.ndarray:
    """
    ### Convert the data to numpy array.

    Parameters
    ----------

    `data` : Union[pd.DataFrame, np.ndarray, list]
    - The data to be converted


    Returns
    -------

    `np.ndarray`
    - The data as numpy array

    ---
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    elif isinstance(data, np.ndarray):
        data = data
    elif isinstance(data, list):
        data = np.array(data)
    else:
        try:
            data = np.array(data)
        except Exception as e:
            raise Exception(
                f"Unable to convert the data to numpy array.\n{e.__class__.__name__}: {e}"
            )

    return data


def train_test_split(
    X: Union[np.ndarray, pd.DataFrame, list],
    Y: Union[np.ndarray, pd.DataFrame, list],
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ### Split the data into training and testing sets.

    Parameters
    ----------

    `X` : np.ndarray
    - The input features of shape (n_samples, n_features)

    `Y` : np.ndarray
    - The output features of shape (n_samples, n_outputs)

    `test_size` : float, optional
    - The size of the testing set, by default 0.2
    - Must be between 0 and 1

    `random_state` : int, optional
    - The random state, by default 0
    - Used to shuffle the data


    Returns
    -------

    `Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
    - The training and testing sets
        [X_train, X_test, Y_train, Y_test]

    - X_train : np.ndarray
        - The input features of the training set
    - X_test : np.ndarray
        - The input features of the testing set
    - Y_train : np.ndarray
        - The output features of the training set
    - Y_test : np.ndarray
        - The output features of the testing set

    ---

    Usage
    -----
    ```python
    from learnML.preprocessing import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    ```

    ---
    """
    X = _get_numpy_data(X)
    Y = _get_numpy_data(Y)

    assert test_size > 0 and test_size < 1, "test_size must be between 0 and 1"
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    assert X.shape[0] > 0, "X must have at least one sample"
    assert Y.shape[0] > 0, "Y must have at least one sample"

    np.random.seed(random_state)

    if X.ndim == 1:
        X = X.reshape(X.shape[0], 1)

    if Y.ndim == 1:
        Y = Y.reshape(Y.shape[0], 1)

    m = X.shape[0]

    # Shuffle the indices of the data points
    indices = np.random.permutation(m)

    # Use the shuffled indices to rearrange X and Y
    X = X[indices, :]
    Y = Y[indices, :]

    # No of samples in the testing set
    testSize = int(m * test_size)

    # Testing set
    X_test = X[:testSize, :]
    Y_test = Y[:testSize, :]

    # Training set
    X_train = X[testSize:, :]
    Y_train = Y[testSize:, :]

    return X_train, X_test, Y_train, Y_test


class KFoldSplit:
    """
    # KFoldSplit

    Class for splitting the data into training and testing
    sets using k-fold cross validation.

    ---

    ## Mathematical Approach

    Let `X` be the input features of shape (n_samples, n_features) and
    `Y` be the output features of shape (n_samples, n_outputs).

    Let `k` be the number of folds.

    Let `m` be the number of samples.

    Then the data is split into `k` folds.

    The `i`th fold is used as the testing set and the remaining folds are used as the training set.

    The data is split into `k` folds using the following steps:

    1. Shuffle the indices of the data points.
    2. Use the shuffled indices to rearrange X and Y.
    3. Split the data into `k` folds.
    4. Repeat steps 1 to 3 `k` times.

    ---

    ## Usage

    To split the data into training and testing sets using k-fold cross validation,
    follow the steps given below:

    1. Import the `KFoldSplit` class from `learnML.preprocessing` module.
    2. Create an instance of the `KFoldSplit` class.
    3. Call the `split` method of the instance created in step 2.
    4. Call the `get_fold` method of the instance created in step 2 to get the training and testing sets.

    ```python
    from learnML.preprocessing import KFoldSplit

    # Create an instance of the KFoldSplit class
    kfold = KFoldSplit(X, Y, k=5)

    # Split the data into training and testing sets
    kfold.split()

    # Get the training and testing sets
    X_train, X_test, Y_train, Y_test = kfold.get_fold(0)
    ```

    ---
    """

    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame, list],
        Y: Union[np.ndarray, pd.DataFrame, list] = None,
        k: int = 5,
    ):
        """
        Parameters
        ----------

        `X` : Union[np.ndarray, pd.DataFrame, list]
        - The input features of shape (n_samples, n_features)

        `Y` : Union[np.ndarray, pd.DataFrame, list], optional
        - The output features of shape (n_samples, n_outputs), by default None
        - If None, then the data is split into k folds without the output features

        `k` : int, optional
        - The number of folds, by default 5
        - Must be greater than 1

        ---
        """
        assert k > 1, "k must be greater than 1"

        X = _get_numpy_data(X)
        if Y is not None:
            Y = _get_numpy_data(Y)

        self._X = X
        self._Y = Y
        self.k = k
        self._foldSize = X.shape[0] // k

        self._splitted_data = None

    def split(self) -> None:
        """
        ### Split the data into training and testing sets using k-fold cross validation.

        Returns
        -------

        None

        ---
        """
        m, n = self._X.shape

        # Shuffle the data
        self._X = self._X[np.random.permutation(m), :]
        self._X = self._X.reshape(m, n)

        if self._Y is not None:
            self._Y = self._Y.reshape(m, 1)
            self._Y = self._Y[np.random.permutation(m), :]
            self._Y = self._Y.reshape(m, 1)

        # Split the data into k folds
        self._splitted_data = []
        for i in range(self.k):
            X_i = self._X[i * self._foldSize : (i + 1) * self._foldSize, :]
            Y_i = None
            if self._Y is not None:
                Y_i = self._Y[i * self._foldSize : (i + 1) * self._foldSize, :]
            self._splitted_data.append((X_i, Y_i))

    def get_fold(
        self, nthFold: int = 0
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        ### Get the training and testing sets for the nth fold.

        Parameters
        ----------

        `nthFold` : int, optional
        - The index of the fold, by default 0
        - Must be between 0 and k - 1

        Returns
        -------

        `Union[
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ]`
        - The training and testing sets
        - If Y is None, then the output is (X_train, X_test)
        - Else, the output is (X_train, X_test, Y_train, Y_test)

        ---
        """
        if self._splitted_data is None:
            self.split()

        nthFold = nthFold % self.k

        X_train = None
        X_test = None
        Y_train = None
        Y_test = None

        for i in range(self.k):
            if i == nthFold:
                X_test, Y_test = self._splitted_data[i]
            else:
                if X_train is None:
                    X_train, Y_train = self._splitted_data[i]
                else:
                    X_train = np.vstack((X_train, self._splitted_data[i][0]))
                    if Y_train is not None:
                        Y_train = np.vstack((Y_train, self._splitted_data[i][1]))

        return (
            (X_train, X_test, Y_train, Y_test)
            if self._Y is not None
            else (X_train, X_test)
        )


class OneLeaveOutSplit:
    """
    # OneLeaveOutSplit

    Class for splitting the data into training and testing
    sets using leave-one-out cross validation.

    ---

    ## Usage

    To use the `OneLeaveOutSplit` class, follow the following steps:

    1. Import the `OneLeaveOutSplit` class from `learnML.preprocessing` module.
    2. Create an instance of the `OneLeaveOutSplit` class.
    3. Call the `split` method of the instance created in step 2.
    4. Call the `get_fold` method of the instance created in step 2 to get the training and testing sets.

    ```python
    from learnML.preprocessing import OneLeaveOutSplit

    # Create an instance of the OneLeaveOutSplit class
    loo = OneLeaveOutSplit(X, Y)

    # Split the data into training and testing sets
    loo.split()

    # Get the training and testing sets
    X_train, X_test, Y_train, Y_test = loo.get_fold(0)
    ```

    ---
    """

    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame, list],
        Y: Union[np.ndarray, pd.DataFrame, list] = None,
    ):
        """
        Parameters
        ----------

        `X` : Union[np.ndarray, pd.DataFrame, list]
        - The input features of shape (n_samples, n_features)

        `Y` : Union[np.ndarray, pd.DataFrame, list], optional
        - The output features of shape (n_samples, n_outputs), by default None
        - If None, then the data is split without the output features

        ---
        """
        X = _get_numpy_data(X)
        if Y is not None:
            Y = _get_numpy_data(Y)

        self._X = X
        self._Y = Y

        self._isRandomized = False

    def randomize(self) -> None:
        """
        ### Randomize the data.

        Returns
        -------

        None

        ---
        """
        m, n = self._X.shape

        # Shuffle the data
        self._X = self._X[np.random.permutation(m), :]
        self._X = self._X.reshape(m, n)

        if self._Y is not None:
            self._Y = self._Y.reshape(m, 1)
            self._Y = self._Y[np.random.permutation(m), :]
            self._Y = self._Y.reshape(m, 1)

        self._isRandomized = True

    def get_fold(
        self, nthFold: int = 0
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        ### Get the training and testing sets for the nth fold.

        Parameters
        ----------

        `nthFold` : int, optional
        - The index of the fold, by default 0
        - Must be between 0 and k - 1

        Returns
        -------

        `Union[
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ]`
        - The training and testing sets
        - If Y is None, then the output is (X_train, X_test)
        - Else, the output is (X_train, X_test, Y_train, Y_test)

        ---
        """
        if self._isRandomized is False:
            self.randomize()

        nthFold = nthFold % self._X.shape[0]

        X_test = self._X[nthFold, :].reshape(1, -1)
        X_train = np.delete(self._X, nthFold, axis=0)

        Y_train = None
        Y_test = None

        if self._Y is not None:
            Y_test = self._Y[nthFold, :].reshape(1, -1)
            Y_train = np.delete(self._Y, nthFold, axis=0)

        return (
            (X_train, X_test, Y_train, Y_test)
            if self._Y is not None
            else (X_train, X_test)
        )
