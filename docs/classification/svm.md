# Class [`LinearSVC`](/learnML/classification/linear_svm.py#L7)

## Introduction

`LinearSVC` (Support Vector Classifier) is a classification algorithm that aims to separate data into two classes by finding a hyperplane that maximizes the margin between them. It's effective for high-dimensional data and can handle small sample sizes efficiently. This algorithm seeks to create a decision boundary by minimizing classification errors and maximizing the margin between the classes.

## Mathematical Approach

`LinearSVC` aims to find a hyperplane that best separates two classes by minimizing a loss function known as the hinge loss. The decision boundary is represented as a linear combination of input features, weights, and an intercept:

```
z = X * W - b
```

Where:

- `X` is the input data matrix of shape `(n_samples, n_features)`.
- `W` is the weight vector of shape `(n_features,)`.
- `b` is the intercept.

The predicted class `y_hat` is determined based on the sign of `z`:

```
y_hat = sign(z)
```

### Hinge Loss

The hinge loss function measures the degree of violation of a sample's classification. For a sample `(x_i, y_i)`, where `x_i` is the input data and `y_i` is the true class label (-1 or 1), the hinge loss is defined as:

```
loss_i = max(0, 1 - y_i * z_i)
```

Where `z_i` is the linear combination for the `i`-th sample. The overall hinge loss for the entire dataset is the sum of individual hinge losses:

```
loss = sum(max(0, 1 - y_i * z_i)) for all samples i
```

### Margin and Support Vectors

The margin is the distance between the decision boundary and the closest data points. The goal is to maximize this margin while minimizing the hinge loss. Support vectors are the data points that are closest to the decision boundary and play a crucial role in defining the hyperplane.

Maximizing the margin is equivalent to minimizing the norm of the weight vector `W`:

```
min (||W|| / 2)
```

### Regularization

Regularization is used to prevent overfitting by penalizing large weights. The regularization term is added to the loss function and is defined as:

```
lambda_ * ||W||^2
```

Where `lambda_` is the regularization parameter.

### Optimization

The goal is to minimize the hinge loss and the regularization term. This is achieved by using gradient descent to iteratively update the weights and intercept. The gradient of the hinge loss function with respect to the weights `W` and intercept `b` is calculated for each sample. For correctly classified samples (`y * z >= 1`), only the regularization term contributes to the gradient. For misclassified samples (`y * z < 1`), both the regularization term and the hinge loss gradient contribute.

The gradients are averaged over all samples and used to update the weights and intercept using the learning rate. This process is repeated for the specified number of iterations.

## Usage

To use the `LinearSVC` model, follow these steps:

1. Import the `LinearSVC` class from the appropriate module.
2. Create an instance of the `LinearSVC` class, specifying hyperparameters.
3. Fit the model to your training data using the `fit` method.
4. Make predictions on new data using the `predict` method.
5. Evaluate the model's performance using the `score` method.

```python
from learnML.classification import LinearSVC

# Create an instance of LinearSVC
model = LinearSVC(learning_rate=0.001, lambda_=0.01, n_iterations=1000)

# Fit the model to training data
model.fit(X_train, Y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Calculate the model's score
model_score = model.score(X_test, Y_test)
```

## Implementation

### Constructor

#### [`__init__(learning_rate: np.float64 = 0.001, lambda_: np.float64 = 0.01, n_iterations: int = 1000, x_scalar: Union[IFeatureEngineering, List[IFeatureEngineering]] = None, debug: bool = True, copy_x: bool = True) -> None`](/learnML/classification/linear_svm.py#L24)

Initialize the LinearSVC model.

- `learning_rate` (np.float64, optional): The learning rate controlling optimization step size.
- `lambda_` (np.float64, optional): The regularization parameter.
- `n_iterations` (int, optional): Number of optimization iterations.
- `x_scalar` (Union[IFeatureEngineering, List[IFeatureEngineering]], optional): Feature engineering for input data.
- `debug` (bool, optional): Debug message printing.
- `copy_x` (bool, optional): Copy input array to avoid modification.

### Methods

#### [`fit(X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = 0.0) -> None`](/learnML/classification/linear_svm.py#L203)

Train the model.

**Mathematical Explanation:**

The training process involves iteratively updating the weights and intercept using gradient descent to minimize the hinge loss cost function. The weights and intercept are updated based on the gradient calculated using the `_gradient` method.

The gradient of the hinge loss function with respect to the weights `W` and intercept `b` is calculated for each sample. For correctly classified samples (`y * z >= 1`), only the regularization term contributes to the gradient. For misclassified samples (`y * z < 1`), both the regularization term and the hinge loss gradient contribute.

The gradients are averaged over all samples and used to update the weights and intercept using the learning rate. This process is repeated for the specified number of iterations.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): Output array of shape `(n_samples,)`.
- `W` (np.ndarray, optional): Weight array of shape `(n_features,)`.
- `b` (np.float64, optional): Intercept value.

#### [`predict(X: np.ndarray) -> np.ndarray`](/learnML/classification/linear_svm.py#L262)

Predict the target data.

**Mathematical Explanation:**

The predicted output is determined by comparing the sign of the linear combination of input features, weights, and intercept with zero.

For each input sample `x_i`, the linear combination `z_i` is computed as `z_i = x_i * W - b`. The predicted class `y_hat_i` for the sample is then given by `y_hat_i = sign(z_i)`. The model predicts class 1 if `z_i` is non-negative and class -1 if `z_i` is negative.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.

#### [`score(X: np.ndarray, Y: np.ndarray, W: np.ndarray = None, b: np.float64 = None) -> np.float64`](/learnML/classification/linear_svm.py#L281)

Compute the cost of the model given input `X`, output `Y`, weights `W`, and intercept `b`.

**Mathematical Explanation:**

The cost of the model is calculated using the hinge loss cost function and regularization term.

The cost is computed as the sum of hinge losses for all samples, normalized by the number of samples. Additionally, the regularization term is added, which penalizes larger values of the weight vector `W`.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray): Output array of shape `(n_samples,)`.
- `W` (np.ndarray, optional): Weight array of shape `(n_features,)`.
- `b` (np.float64, optional): Intercept value.

### Private Methods

#### [`_y_hat(X: np.ndarray, W: np.ndarray, b: np.float64) -> np.ndarray`](/learnML/classification/linear_svm.py#L63)

Calculate predicted value of `y` given input `X`, weights `W`, and intercept `b`.

**Mathematical Explanation:**

The predicted value `y_hat` is calculated by computing the linear combination of input features, weights, and intercept.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `W` (np.ndarray): Weight array of shape `(n_features,)`.
- `b` (np.float64): Intercept value.

#### [`_cost(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64) -> np.float64`](/learnML/classification/linear_svm.py#L83)

Compute cost of the model given input `X`, output `y`, weights `W`, and intercept `b`.

**Mathematical Explanation:**

The cost of the model is calculated using the hinge loss cost function and regularization term.

The cost is computed as the sum of hinge losses for all samples, normalized by the number of samples. Additionally, the regularization term is added, which penalizes larger values of the weight vector `W`.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `y` (np.ndarray): Output array of shape `(n_samples,)`.
- `W` (np.ndarray): Weight array of shape `(n_features,)`.
- `b` (np.float64): Intercept value.

#### [`_gradient(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64) -> Union[np.ndarray, np.float64]`](/learnML/classification/linear_svm.py#L120)

Compute gradient of the model with respect to weights and intercept.

**Mathematical Explanation:**

The gradient of the hinge loss function with respect to the weights `W` and intercept `b` is calculated for each sample. For correctly classified samples (`y * z >= 1`), only the regularization term contributes to the gradient. For misclassified samples (`y * z < 1`), both the regularization term and the hinge loss gradient contribute.

The gradient of the hinge loss with respect to the weights is calculated using conditional logic based on whether a sample is misclassified or correctly classified. For misclassified samples, the gradient of the hinge loss contributes. For correctly classified samples, the gradient of the regularization term contributes.

The gradients are averaged over all samples and used to update the weights and intercept using the learning rate. This process is repeated for the specified number of iterations.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `y` (np.ndarray): Output array of shape `(n_samples,)`.
- `W` (np.ndarray): Weight array of shape `(n_features,)`.
- `b` (np.float64): Intercept value.

#### [`_validate_data(X: np.ndarray, Y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]`](/learnML/classification/linear_svm.py#L166)

Validate and preprocess input and output data.

**Mathematical Explanation:**

This method preprocesses input and output data, applies feature engineering, and transforms class labels into -1 and 1.

- `X` (np.ndarray): Input array of shape `(n_samples, n_features)`.
- `Y` (np.ndarray, optional): Output array of shape `(n_samples,)`.

## Notes

- The `LinearSVC` class implements a linear support vector classifier for binary classification tasks.
- It inherits from the `IRegression` interface and provides implementations for the `fit`, `predict`, and `score` methods.
- Feature engineering can be applied to the input data using the `x_scalar` parameter.
- Private methods `_y_hat`, `_cost`, `_gradient`, and `_validate_data` are used internally for different aspects of the SVM algorithm.
- The hinge loss is used to measure classification error, and the margin is the distance between the decision boundary and the closest data points (support vectors).
- The model seeks to minimize the hinge loss while also maximizing the margin to achieve an effective classification boundary.
