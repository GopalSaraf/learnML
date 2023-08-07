import numpy as np
from typing import Tuple


def train_test_split(
    X: np.ndarray, Y: np.ndarray, test_size: float = 0.2, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    X : np.ndarray
        The input features of shape (n_samples, n_features)
    Y : np.ndarray
        The output features of shape (n_samples, n_outputs)
    test_size : float, optional
        The size of the testing set, by default 0.2
    random_state : int, optional
        The random state, by default 0

    Returns
    -------
    tuple
        The training and testing sets
        X_train, X_test, Y_train, Y_test
    """
    assert test_size > 0 and test_size < 1, "test_size must be between 0 and 1"
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    assert X.shape[0] > 0, "X must have at least one sample"
    assert Y.shape[0] > 0, "Y must have at least one sample"

    np.random.seed(random_state)

    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)

    if len(Y.shape) == 1:
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
    """Class for splitting the data into training and testing sets using k-fold cross validation."""

    def __init__(self, X: np.ndarray, Y: np.ndarray = None, k: int = 5):
        """
        Initialize the KFoldSplit class.

        Parameters
        ----------
        data : np.ndarray
            The data to be split
        k : int, optional
            The number of folds, by default 5
        """
        assert k > 1, "k must be greater than 1"

        self._X = X
        self._Y = Y
        self.k = k
        self._foldSize = X.shape[0] // k

        self._splitted_data = None

    def split(self) -> None:
        """
        Split the data into training and testing sets using k-fold cross validation.

        Returns
        -------
        None
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

    def get_fold(self, nthFold: int = 0) -> tuple:
        """
        Get the training and testing sets for the nth fold.

        Parameters
        ----------
        nthFold : int, optional
            The fold number, by default 0

        Returns
        -------
        tuple
            The training and testing sets
            X_train, X_test, Y_train, Y_test (if Y is not None)
            X_train, X_test (if Y is None)
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
    """Class for splitting the data into training and testing sets using leave-one-out cross validation."""

    def __init__(self, X: np.ndarray, Y: np.ndarray = None):
        """
        Initialize the OneLeaveOutSplit class.

        Parameters
        ----------
        X : np.ndarray
            The input features
        Y : np.ndarray, optional
            The output features, by default None
        """
        self._X = X
        self._Y = Y

        self._isRandomized = False

    def randomize(self) -> None:
        """
        Randomize the data.

        Returns
        -------
        None
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

    def get_fold(self, nthFold: int = 0) -> tuple:
        """
        Get the training and testing sets for the nth fold.

        Parameters
        ----------
        nthFold : int, optional
            The fold number, by default 0

        Returns
        -------
        tuple
            The training and testing sets
            X_train, X_test, Y_train, Y_test (if Y is not None)
            X_train, X_test (if Y is None)
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
