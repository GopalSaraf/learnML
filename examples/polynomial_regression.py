import numpy as np
import matplotlib.pyplot as plt

# Add the path of learnML to sys.path
import sys

sys.path.append("../learnML")

from learnML.preprocessing import ZScoreNormalization, PolynomialFeatures
from learnML.regression import MultipleLinearRegression


X = np.arange(0, 20, 1)
y = 1 + X**2


X = X.reshape(-1, 1)

# feature engineering
feature_engineering = PolynomialFeatures(X, degree=[2, 3])
X_poly = feature_engineering.transform()

# feature scaling
X = ZScoreNormalization(X).fit_transform()
X_poly = ZScoreNormalization(X_poly).fit_transform()

multiple_linear_regression_model = MultipleLinearRegression(num_iterations=100000)
multiple_linear_regression_model.fit(X, y)

print(
    f"Multiple Linear Regression Model Weights: {multiple_linear_regression_model.get_weights()}, Intercept: {multiple_linear_regression_model.get_intercept()}"
)

poly_linear_regression_model = MultipleLinearRegression(num_iterations=100000)
poly_linear_regression_model.fit(X_poly, y)

print(
    f"Polynomial Linear Regression Model Weights: {poly_linear_regression_model.get_weights()}, Intercept: {poly_linear_regression_model.get_intercept()}"
)

# plot the data
plt.plot(X, multiple_linear_regression_model.predict(X), color="red")
plt.plot(X, poly_linear_regression_model.predict(X_poly), color="green")
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.legend(["Multiple Linear Regression", "Polynomial Regression"])
plt.title("Multiple Linear Regression vs Polynomial Regression")
plt.show()
