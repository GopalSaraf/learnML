import numpy as np
import matplotlib.pyplot as plt

# Add the path of learnML to sys.path
import sys

sys.path.append("../learnML")

from learnML.regression import UnivariateLinearRegression

# Generate data points
X = np.linspace(0, 10, 100)
Y = 2 * X + 1 + np.random.normal(0, 1, 100)

model = UnivariateLinearRegression(learning_rate=0.001, num_iterations=10000)
model.fit(X, Y)

print(f"After training: w = {model.get_weight()}, b = {model.get_intercept()}")


def plot():
    # Plot the data points
    plt.figure()
    plt.scatter(X, Y)
    plt.plot(X, model.predict_all(X))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data Points and Linear Regression")
    plt.draw()
    plt.pause(0.01)

    # Plot the cost function vs iterations
    plt.figure()
    plt.plot(model.get_J_history())
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function vs Iterations")
    plt.draw()
    plt.pause(0.01)

    # Plot the parameters vs iterations
    plt.figure()
    plt.plot(model.get_p_history())
    plt.xlabel("Iteration")
    plt.ylabel("Parameter")
    plt.legend(["w", "b"])
    plt.title("Parameters vs Iterations")
    plt.draw()
    plt.pause(0.01)

    # Plot the cost function vs parameters
    plt.figure()
    w = np.linspace(-100, 100, 1000)
    b = np.linspace(-100, 100, 1000)
    W, B = np.meshgrid(w, b)
    J = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = model.cost(X, Y, W[i, j], B[i, j])

    plt.contourf(W, B, J, 100)
    plt.xlabel("w")
    plt.ylabel("b")
    plt.colorbar()
    plt.title("Cost Function vs Parameters")
    plt.draw()

    plt.show()


want_plot = input("Do you want to plot the results? (y/n): ")
if want_plot.lower() == "y":
    plot()
