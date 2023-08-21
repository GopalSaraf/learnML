from abc import ABC, abstractmethod
import numpy as np


class IModel(ABC):
    """
    # IModel

    Interface for model classes.

    The `IModel` abstract class serves as a foundational interface for all model classes within the `learnML` library. By providing a consistent structure, it ensures that each model adheres to a common set of methods and behaviors. Models that inherit from `IModel` are expected to implement the `fit`, `predict`, and `score` methods, allowing users to seamlessly interchange and evaluate different models in their machine learning pipelines.

    ---

    ## Usage

    ```python
    from learnML.interfaces import IModel

    # Define a custom model class that inherits from IModel
    class CustomModel(IModel):
        def fit(self, X, Y):
            # Implementation of the fit method
            pass

        def predict(self, X):
            # Implementation of the predict method
            pass

        def score(self, X, Y):
            # Implementation of the score method
            pass
    ```

    ---
    """

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        ### Fit the model to the data.

        Parameters
        ----------

        `X` : np.ndarray
        - The array like object containing the input
            data of shape (n_samples, n_features)

        `Y` : np.ndarray
        - The array like object containing the output
            data of shape (n_samples, n_targets) or (n_samples,)


        Returns
        -------

        None

        ---
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ### Predict the output given the input.

        Parameters
        ----------

        `X` : np.ndarray
        - The array like object containing the input
            data of shape (n_samples, n_features)


        Returns
        -------
        `np.ndarray`
        - The array like object containing the output
            data of shape (n_samples, n_targets) or (n_samples,)

        ---
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        """
        ### Calculate the score of the model.

        Parameters
        ----------

        `X` : np.ndarray
        - The array like object containing the input
            data of shape (n_samples, n_features)

        `Y` : np.ndarray
        - The array like object containing the output
            data of shape (n_samples, n_targets) or (n_samples,)


        Returns
        -------

        `np.float64`
        - The score of the model

        ---
        """
        pass

    def __get_numpy_array(self, X: np.ndarray) -> np.ndarray:
        """
        ### Convert the input to numpy array.

        Parameters
        ----------

        `X` : np.ndarray
        - The array like object containing the input
            data of shape (n_samples, n_features)


        Returns
        -------

        `np.ndarray`
        - The array like object containing the input
            data of shape (n_samples, n_features)

        ---
        """
        if not isinstance(X, np.ndarray):
            return np.array(X)
        return X
