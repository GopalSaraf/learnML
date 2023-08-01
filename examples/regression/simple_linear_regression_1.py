import numpy as np
from learnML.regression import UnivariateLinearRegression

from plt import plot

# Input data for univariate regression
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y_train = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10])

# Train the model
model = UnivariateLinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(x_train, y_train)

x_test = np.array([10, 11, 12, 13, 14, 15])
y_test = np.array([11, 13, 14, 15, 17, 18])


# Predict the output
y_pred = model.predict(x_test)

# Print the predicted output
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)


plot(
    model=model,
    subtitle="Univariate Linear Regression",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
)
