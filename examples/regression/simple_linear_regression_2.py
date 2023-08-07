import pandas as pd
import numpy as np

from learnML.regression import UnivariateLinearRegression
from learnML.preprocessing import train_test_split

from plt import plot

# Load the data
data = pd.read_csv("data/salary_data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train the model
model = UnivariateLinearRegression(learning_rate=0.001, num_iterations=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)

plot(
    model=model,
    subtitle="Univariate Linear Regression",
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
)
