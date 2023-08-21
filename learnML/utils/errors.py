import numpy as np


class Errors:
    """
    A class to measure the accuracy of a model using Errors.
    """

    @staticmethod
    def mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the mean squared error of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The mean squared error of model
        """
        y_pred = y_pred.reshape(-1)
        y_test = y_test.reshape(-1)
        return np.sum((y_test - y_pred) ** 2) / len(y_test)

    @staticmethod
    def mean_absolute_error(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the mean absolute error of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The mean absolute error of model
        """
        y_pred = y_pred.reshape(-1)
        y_test = y_test.reshape(-1)
        return np.sum(np.abs(y_test - y_pred)) / len(y_test)

    @staticmethod
    def mean_squared_log_error(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the mean squared log error of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The mean squared log error of model
        """
        y_pred = y_pred.reshape(-1)
        y_test = y_test.reshape(-1)
        return np.sum((np.log(y_pred + 1) - np.log(y_test + 1)) ** 2) / len(y_test)


@staticmethod
def mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
    """
    Returns the mean squared error of model.

    Parameters
    ----------
    y_pred : np.ndarray
        The model prediction array of shape (n_samples, ) or (1, n_samples)
    y_test : np.ndarray
        The testing array of shape (n_samples, ) or (1, n_samples)

    Returns
    -------
    np.float64
        The mean squared error of model
    """
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    return np.sum((y_test - y_pred) ** 2) / len(y_test)


@staticmethod
def mean_absolute_error(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
    """
    Returns the mean absolute error of model.

    Parameters
    ----------
    y_pred : np.ndarray
        The model prediction array of shape (n_samples, ) or (1, n_samples)
    y_test : np.ndarray
        The testing array of shape (n_samples, ) or (1, n_samples)

    Returns
    -------
    np.float64
        The mean absolute error of model
    """
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    return np.sum(np.abs(y_test - y_pred)) / len(y_test)


@staticmethod
def mean_squared_log_error(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
    """
    Returns the mean squared log error of model.

    Parameters
    ----------
    y_pred : np.ndarray
        The model prediction array of shape (n_samples, ) or (1, n_samples)
    y_test : np.ndarray
        The testing array of shape (n_samples, ) or (1, n_samples)

    Returns
    -------
    np.float64
        The mean squared log error of model
    """
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    return np.sum((np.log(y_pred + 1) - np.log(y_test + 1)) ** 2) / len(y_test)
