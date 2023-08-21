from abc import ABC, abstractmethod
import numpy as np


class IFeatureEngineering(ABC):
    """
    # IFeatureEngineering

    Interface for feature engineering classes.

    `IFeatureEngineering` is an abstract interface that defines methods for performing feature engineering on input data. Feature engineering involves transforming and preprocessing raw input data to create more informative and suitable features for machine learning models.

    ---

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

    ---
    """

    @abstractmethod
    def __init__(self, data: np.ndarray) -> None:
        """
        Parameters
        ----------

        `data` : np.ndarray
        - The input array of shape (n_samples, n_features)

        ---
        """

        self._data = data

    @abstractmethod
    def fit(self, data: np.ndarray = None) -> None:
        """
        ### Fit the feature engineer to data

        Parameters
        ----------

        `data` : np.ndarray, optional
        - The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)


        Returns
        -------
        None

        ---
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        ### Transform data using feature engineer

        Parameters
        ----------

        `data` : np.ndarray, optional
        - The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)


        Returns
        -------

        `np.ndarray`
        - The transformed data of shape (n_samples, n_features)

        ---
        """
        pass

    @abstractmethod
    def fit_transform(self, data: np.ndarray = None) -> np.ndarray:
        """
        ### Fit the feature engineer with data and transform with it

        Parameters
        ----------

        `data` : np.ndarray, optional
        - The input array of shape (n_samples, n_features),
            by default None (uses the input array passed in the constructor)


        Returns
        -------

        `np.ndarray`
        - The transformed data of shape (n_samples, n_features)

        ---
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        ### Convert the data back to the original representation

        Parameters
        ----------

        `data` : np.ndarray
        - The input array of shape (n_samples, n_features)


        Returns
        -------

        `np.ndarray`
        - The transformed data of shape (n_samples, n_features)

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
