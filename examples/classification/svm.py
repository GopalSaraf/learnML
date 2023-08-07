import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from learnML.classification import LinearSVC
from learnML.preprocessing import train_test_split, ZScoreNormalization
from learnML.utils.accuracy import Metrics

data = pd.read_csv("data/svm_data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

model = LinearSVC(
    learning_rate=0.001,
    lambda_param=0.01,
    num_iterations=10000,
)

model.fit(X, y)

print("After training:")
print("Weights: ", model.get_weights())
print("Intercept: ", model.get_intercept())

# print("After training:")
# print("Weights: ", model.w)
# print("Intercept: ", model.b)


def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    weights = model.get_weights()
    intercept = model.get_intercept()

    # weights = model.w
    # intercept = model.b

    x1_1 = get_hyperplane_value(x0_1, weights, intercept, 0)
    x1_2 = get_hyperplane_value(x0_2, weights, intercept, 0)

    x1_1_m = get_hyperplane_value(x0_1, weights, intercept, -1)
    x1_2_m = get_hyperplane_value(x0_2, weights, intercept, -1)

    x1_1_p = get_hyperplane_value(x0_1, weights, intercept, 1)
    x1_2_p = get_hyperplane_value(x0_2, weights, intercept, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


visualize_svm()
