import numpy as np


class Metrics:
    """
    A class to measure the accuracy of a model.
    """

    @staticmethod
    def confusion_matrix(y_pred: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Returns the confusion matrix of the model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.ndarray
            The confusion matrix of the model
        """
        if len(y_pred.shape) == 2:
            y_pred = y_pred.reshape(-1)

        if len(y_test.shape) == 2:
            y_test = y_test.reshape(-1)

        assert y_pred.shape == y_test.shape, (
            f"y_pred and y_test should contain same number of samples. "
            f"y_pred contains {y_pred.shape[0]} whereas y_test contain {y_test.shape[0]} samples"
        )

        unique = np.unique(y_test)
        n = len(unique)

        matrix = np.zeros((n, n), dtype=int)

        for i in range(len(y_test)):
            matrix[y_test[i], y_pred[i]] += 1

        return matrix

    @staticmethod
    def accuracy_score(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the accuracy score of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The accuracy of model
        """
        matrix = Metrics.confusion_matrix(y_pred, y_test)
        return np.sum(np.diag(matrix)) / np.sum(matrix)

    @staticmethod
    def precision_score(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the precision score of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The precision of model
        """
        matrix = Metrics.confusion_matrix(y_pred, y_test)
        return matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])

    @staticmethod
    def recall_score(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the recall score of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The recall of model
        """
        matrix = Metrics.confusion_matrix(y_pred, y_test)
        return matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])

    @staticmethod
    def f1_score(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the f1 score of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The f1 score of model
        """
        precision = Metrics.precision_score(y_pred, y_test)
        recall = Metrics.recall_score(y_pred, y_test)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def r2_score(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
        """
        Returns the r2 score of model.

        Parameters
        ----------
        y_pred : np.ndarray
            The model prediction array of shape (n_samples, ) or (1, n_samples)
        y_test : np.ndarray
            The testing array of shape (n_samples, ) or (1, n_samples)

        Returns
        -------
        np.float64
            The r2 score of model
        """
        y_pred = y_pred.reshape(-1)
        y_test = y_test.reshape(-1)
        mean = np.mean(y_test)
        return 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - mean) ** 2)

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
