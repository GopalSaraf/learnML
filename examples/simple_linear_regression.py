import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data/salary_data.csv")

# Add the path of learnML to sys.path
import sys

sys.path.append("../")

from learnML.regression import UnivariateLinearRegression
from learnML.preprocessing import TrainTestSplit

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
    X, y, test_size=1 / 3
)


# Train the model
model = UnivariateLinearRegression(learning_rate=0.001, num_iterations=10000)
model.fit(X_train, y_train)

# Visualising the Training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, model.predict_all(X_train), color="blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, model.predict_all(X_train), color="blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
