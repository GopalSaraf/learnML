import pandas as pd
import numpy as np


class TrainTestSplit:
    """Class for splitting the data into training and testing sets."""

    @staticmethod
    def k_fold_pandas_split(
        X: pd.DataFrame, Y: pd.DataFrame, k: int = 5, nthFold: int = 0
    ) -> tuple:
        """
        Split the data into training and testing sets using k-fold cross validation.

        Parameters
        ----------
        X : pd.DataFrame
            The input features
        Y : pd.DataFrame
            The output features
        k : int, optional
            The number of folds, by default 5
        nthFold : int, optional
            The fold number to use as the testing set, by default 0

        Returns
        -------
        tuple
            The training and testing sets
            X_train, X_test, Y_train, Y_test
        """
        assert k > 1, "k must be greater than 1"

        m, n = X.shape
        nthFold = nthFold % k

        # Shuffle the data
        X = X.sample(frac=1, random_state=k).reset_index(drop=True)
        Y = Y.sample(frac=1, random_state=k).reset_index(drop=True)

        # No of samples in each fold
        foldSize = m // k

        # Index of the start and end of the testing set
        start = nthFold * foldSize
        end = start + foldSize

        # Testing set
        X_test = X.iloc[start:end, :].reset_index(drop=True)
        Y_test = Y.iloc[start:end, :].reset_index(drop=True)

        # Training set
        X_train = pd.concat([X.iloc[:start, :], X.iloc[end:, :]]).reset_index(drop=True)
        Y_train = pd.concat([Y.iloc[:start, :], Y.iloc[end:, :]]).reset_index(drop=True)

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def k_fold_split(
        X: np.ndarray, Y: np.ndarray, k: int = 5, nthFold: int = 0
    ) -> tuple:
        """
        Split the data into training and testing sets using k-fold cross validation.

        Parameters
        ----------
        X : np.ndarray
            The input features
        Y : np.ndarray
            The output features
        k : int, optional
            The number of folds, by default 5
        nthFold : int, optional
            The fold number to use as the testing set, by default 0

        Returns
        -------
        tuple
            The training and testing sets
            X_train, X_test, Y_train, Y_test
        """
        assert k > 1, "k must be greater than 1"

        m, n = X.shape
        nthFold = nthFold % k
        Y = Y.reshape(m, 1)

        # Shuffle the data
        X = X[np.random.permutation(m), :]
        Y = Y[np.random.permutation(m), :]
        X = X.reshape(m, n)
        Y = Y.reshape(m, 1)

        # No of samples in each fold
        foldSize = m // k

        # Index of the start and end of the testing set
        start = nthFold * foldSize
        end = start + foldSize

        # Testing set
        X_test = X[start:end, :]
        Y_test = Y[start:end, :]

        # Training set
        X_train = np.concatenate((X[:start, :], X[end:, :]), axis=0)
        Y_train = np.concatenate((Y[:start, :], Y[end:, :]), axis=0)

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def train_test_pandas_split(
        X: pd.DataFrame, Y: pd.DataFrame, test_size: float = 0.2, random_state: int = 0
    ) -> tuple:
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        X : pd.DataFrame
            The input features
        Y : pd.DataFrame
            The output features
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
        m, n = X.shape

        # Shuffle the data
        X = X.sample(frac=1, random_state=random_state).reset_index(drop=True)
        Y = Y.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # No of samples in the testing set
        testSize = int(m * test_size)

        # Testing set
        X_test = X.iloc[:testSize, :].reset_index(drop=True)
        Y_test = Y.iloc[:testSize, :].reset_index(drop=True)

        # Training set
        X_train = X.iloc[testSize:, :].reset_index(drop=True)
        Y_train = Y.iloc[testSize:, :].reset_index(drop=True)

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def train_test_split(
        X: np.ndarray, Y: np.ndarray, test_size: float = 0.2, random_state: int = 0
    ) -> tuple:
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

    @staticmethod
    def leave_one_out_pandas_split(
        X: pd.DataFrame, Y: pd.DataFrame, nthSample: int = 0
    ) -> tuple:
        """
        Split the data into training and testing sets using leave one out cross validation.

        Parameters
        ----------
        X : pd.DataFrame
            The input features
        Y : pd.DataFrame
            The output features
        nthSample : int, optional
            The sample number to use as the testing set, by default 0

        Returns
        -------
        tuple
            The training and testing sets
            X_train, X_test, Y_train, Y_test
        """
        m, n = X.shape
        nthSample = nthSample % m

        # Testing set
        X_test = X.iloc[nthSample, :].reset_index(drop=True)
        Y_test = Y.iloc[nthSample, :].reset_index(drop=True)

        # Training set
        X_train = X.drop(nthSample).reset_index(drop=True)
        Y_train = Y.drop(nthSample).reset_index(drop=True)

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def leave_one_out_split(X: np.ndarray, Y: np.ndarray, nthSample: int = 0) -> tuple:
        """
        Split the data into training and testing sets using leave one out cross validation.

        Parameters
        ----------
        X : np.ndarray
            The input features
        Y : np.ndarray
            The output features
        nthSample : int, optional
            The sample number to use as the testing set, by default 0

        Returns
        -------
        tuple
            The training and testing sets
            X_train, X_test, Y_train, Y_test
        """
        m, n = X.shape
        nthSample = nthSample % m

        # Testing set
        X_test = X[nthSample, :]
        Y_test = Y[nthSample, :]

        # Training set
        X_train = np.delete(X, nthSample, axis=0)
        Y_train = np.delete(Y, nthSample, axis=0)

        return X_train, X_test, Y_train, Y_test
