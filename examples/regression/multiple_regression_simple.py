import numpy as np
import matplotlib.pyplot as plt

from learnML.regression import MultipleLinearRegression


X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])


model = MultipleLinearRegression(learning_rate=10.0e-10, num_iterations=10000)
model.fit(X, y)

print(f"After training: w = {model.get_weights()}, b = {model.get_intercept()}")

X_test = np.array([[2104, 5, 1, 45]])
y_pred = model.predict(X_test)

print(f"Prediction for {X_test} : {y_pred[0]}")


plt.plot(model.get_cost_history())
plt.title("Cost vs. iteration")
plt.ylabel("Cost")
plt.xlabel("iteration step")
plt.show()
