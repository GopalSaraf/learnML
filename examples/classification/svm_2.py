import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from learnML.classification import LinearSVC
from learnML.preprocessing import train_test_split, ZScoreNormalization

# Load the data
dataset = pd.read_csv("data/social_ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training set:")
print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
print()
print("Testing set:")
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)
print()

# Normalize the data
X_scalar = ZScoreNormalization(X_train)
X_scalar.fit()

# Train the model
model = LinearSVC(
    learning_rate=0.001,
    lambda_param=0.01,
    num_iterations=5000,
    X_scalar=X_scalar,
)

model.fit(X_train, y_train)

print()
print("Training completed!")
print("Weights:", model.get_weights())
print("Intercept:", model.get_intercept())
print()

# Predict
y_pred = model.predict(X_test)
y_test = y_test.reshape(-1)

print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

print()
print("Accuracy:", np.sum(y_pred == y_test) / len(y_test))
print()
print("Confusion matrix:")
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))


plt.plot(model.get_cost_history())
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost per iteration")
plt.show()
