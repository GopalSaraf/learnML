""" Graph plotter helper """

import matplotlib.pyplot as plt

# Add the path of learnML to sys.path
import sys

sys.path.append("../")

from learnML.interfaces import IModel


def createFigure(subtitle):
    # Create a figure with two subplots side by side
    fig, (ax_upper, ax_lower) = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(subtitle)

    return ax_upper[0], ax_upper[1], ax_lower[0], ax_lower[1]


def pltTrainSet(ax, x_train, y_train, y_pred):
    # Plot the Training set in the first subplot
    ax.scatter(x_train, y_train, color="red", label="Training set")
    ax.plot(x_train, y_pred, color="blue")
    ax.set_title("Training set")
    ax.set_xlabel("x_train")
    ax.set_ylabel("y_train")
    ax.legend()


def pltTestSet(ax, x_test, y_test, y_pred):
    # Plot the Test set in the second subplot
    ax.scatter(x_test, y_test, color="red", label="Test set")
    ax.plot(x_test, y_pred, color="blue")
    ax.set_title("Salary vs Experience (Test set)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.legend()


def pltCostVsIter(ax, j_history):
    # Plot the Cost vs Iterations in the third subplot
    ax.plot(j_history)
    ax.set_title("Cost vs Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")


def pltWeightsVsIter(ax, p_history):
    # Plot the weights vs Iterations in the fourth subplot
    ax.plot(p_history)
    ax.set_title("Weights vs Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Weights")
    ax.legend(["weight", "intercept"])


def plot(model: IModel, subtitle, x_train, y_train, x_test=None, y_test=None):
    ax1, ax2, ax3, ax4 = createFigure(subtitle)
    pltTrainSet(ax1, x_train, y_train, model.predict(x_train))

    if x_test is not None and y_test is not None:
        pltTestSet(ax2, x_test, y_test, model.predict(x_test))

    pltCostVsIter(ax3, model.get_cost_history())
    pltWeightsVsIter(ax4, model.get_parameter_history())

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
